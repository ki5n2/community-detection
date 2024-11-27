#%%
'''IMPORTS'''
import torch
import numpy as np
import torch.nn as nn
import matplotlib.pyplot as plt
import torch.nn.functional as F
import torch_geometric.transforms as T

from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from torch_geometric.nn import GATConv
from torch_geometric.datasets import Planetoid
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score


#%%
class CDBNE(nn.Module):
    def __init__(self, input_dim, hidden_dim, embedding_dim, n_clusters, alpha=1.0):
        super(CDBNE, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.embedding_dim = embedding_dim
        self.n_clusters = n_clusters
        self.alpha = alpha
        
        # Encoder with proper initialization
        self.encoder_gat1 = GATConv(input_dim, hidden_dim)
        self.encoder_gat2 = GATConv(hidden_dim, embedding_dim)
        
        # Decoder with proper initialization
        self.decoder_gat1 = GATConv(embedding_dim, hidden_dim)
        self.decoder_gat2 = GATConv(hidden_dim, input_dim)
        
        # Clustering layer initialized with small random values
        self.cluster_layer = nn.Parameter(torch.randn(n_clusters, embedding_dim) * 0.1)
        # torch.nn.init.xavier_normal_(self.cluster_layer.data)

    def encode(self, x, edge_index):
        h1 = F.relu(self.encoder_gat1(x, edge_index))
        z = self.encoder_gat2(h1, edge_index)
        return z

    def decode(self, z, edge_index):
        h1 = F.relu(self.decoder_gat1(z, edge_index))
        x_hat = self.decoder_gat2(h1, edge_index)
        return x_hat

    def forward(self, x, edge_index):
        # Apply layer normalization to input
        x = F.normalize(x, p=2, dim=1)
        
        # Get embeddings
        z = self.encode(x, edge_index)
        
        # Apply normalization to embeddings
        z = F.normalize(z, p=2, dim=1)
        
        # Get reconstruction and cluster probabilities
        x_hat = self.decode(z, edge_index)
        q = self.get_cluster_prob(z)
        
        return z, x_hat, q

    def get_cluster_prob(self, z):
        z_norm = torch.sum(torch.square(z), dim=1, keepdim=True)
        cluster_norm = torch.sum(torch.square(self.cluster_layer), dim=1, keepdim=True)
        dist = z_norm + cluster_norm.t() - 2 * torch.mm(z, self.cluster_layer.t())
        q = 1.0 / (1.0 + (dist / self.alpha))
        q = (q + 1e-7) ** ((self.alpha + 1.0) / 2.0)
        q = (q.t() / torch.sum(q, dim=1)).t()
        return q

    # def loss_modularity(self, z, adj):
    #     adj = adj.to(z.device)
    #     degrees = torch.sum(adj, dim=1)
    #     m = torch.sum(adj) / 2.0
    #     B = adj - torch.outer(degrees, degrees) / (2.0 * m)
    #     H = F.softmax(torch.mm(z, self.cluster_layer.t()), dim=1)
    #     Q = torch.trace(torch.mm(torch.mm(H.t(), B), H)) / (4.0 * m)
    #     return -Q
    def loss_modularity(self, z, adj):
        """
        Calculate modularity loss according to equation (11) in the paper
        Q = 1/4m * Tr(H^T B H)
        where B = A - dd^T/2m is the modularity matrix
        """
        # Ensure input is float and on correct device
        adj = adj.to(z.device).float()
        
        # Calculate degrees (d) and total number of edges (m)
        degrees = torch.sum(adj, dim=1)
        m = torch.sum(adj) / 2.0
        
        # Normalize degrees for numerical stability
        degrees = degrees / (2.0 * m)
        
        # Calculate modularity matrix B = A - dd^T/(2m)
        # Using outer product for dd^T
        B = adj - torch.outer(degrees, degrees) * (2.0 * m)
        
        # Calculate community assignments using softmax
        # H shape: (n_nodes, n_clusters)
        H = F.softmax(torch.mm(z, self.cluster_layer.t()), dim=1)
        
        # Calculate modularity Q = 1/4m * Tr(H^T B H)
        # First calculate H^T B H
        HtB = torch.mm(H.t(), B)  # (n_clusters, n_nodes)
        HtBH = torch.mm(HtB, H)   # (n_clusters, n_clusters)
        Q = torch.trace(HtBH) / (4.0 * m)
        
        # Print debug information
        if torch.isnan(Q) or torch.isinf(Q):
            print("Warning: Q is nan or inf")
            print(f"m: {m}")
            print(f"degrees min/max: {degrees.min()}/{degrees.max()}")
            print(f"B min/max: {B.min()}/{B.max()}")
            print(f"H min/max: {H.min()}/{H.max()}")
            print(f"Q: {Q}")
        
        # Add small epsilon to avoid numerical instability
        Q = Q + 1e-10
    
        return -Q  # Return negative since we want to maximize modularity

    def loss_reconstruction(self, x, x_hat, edge_index):
        loss_attr = F.mse_loss(x_hat, x)
        z = self.encode(x, edge_index)
        adj_pred = torch.sigmoid(torch.mm(z, z.t()))
        adj_true = torch.zeros((x.size(0), x.size(0)), device=x.device)
        adj_true[edge_index[0], edge_index[1]] = 1
        loss_struct = F.binary_cross_entropy(adj_pred, adj_true)
        return loss_attr + loss_struct


#%%
class CDBNE_Trainer:
    def __init__(self, model, optimizer, device, beta=0.1, gamma=1.0, update_interval=1):
        self.model = model
        self.optimizer = optimizer
        self.device = device
        self.beta = beta
        self.gamma = gamma
        self.update_interval = update_interval
        # Learning rate scheduler 추가
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=10, verbose=True
        )
        
    def calculate_target_dist(self, q):
        f = torch.sum(q, dim=0)
        numerator = q ** 2 / f
        denominator = torch.sum(numerator, dim=1, keepdim=True)
        p = numerator / denominator
        return p

    def train_step(self, data, target_dist=None):
        self.model.train()
        self.optimizer.zero_grad()
        
        z, x_hat, q = self.model(data.x, data.edge_index)
        loss_recon = self.model.loss_reconstruction(data.x, x_hat, data.edge_index)
        loss_mod = self.model.loss_modularity(z, data.adj)
        
        loss_cluster = 0
        if target_dist is not None:
            loss_cluster = torch.sum(target_dist * torch.log(target_dist / (q + 1e-7)))
        
        loss = loss_recon - self.beta * loss_mod + self.gamma * loss_cluster
        
        loss.backward()
        self.optimizer.step()
        
        return loss.item(), q

    def pretrain(self, data, epochs):
        self.model.train()
        best_loss = float('inf')
        
        for epoch in range(epochs):
            self.optimizer.zero_grad()
            
            # Forward pass
            z, x_hat, _ = self.model(data.x, data.edge_index)
            
            # Calculate losses with scale normalization
            loss_recon = self.model.loss_reconstruction(data.x, x_hat, data.edge_index)
            loss_mod = self.model.loss_modularity(z, data.adj)
            
            # Scale normalization
            if epoch == 0:
                self.recon_scale = loss_recon.item()
                self.mod_scale = abs(loss_mod.item()) + 1e-8
            
            loss = (loss_recon / self.recon_scale) - self.beta * (loss_mod / self.mod_scale)
            
            # Backward pass with gradient clipping
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            
            # Learning rate scheduling
            self.scheduler.step(loss)
            
            # Print progress
            if (epoch + 1) % 100 == 0:
                print(f'Pretrain Epoch {epoch+1}: Loss = {loss.item():.4f} '
                      f'(Recon = {loss_recon.item():.4f}, Mod = {loss_mod.item():.4f})')
            
            # Early stopping
            if loss.item() < best_loss:
                best_loss = loss.item()
                best_z = z.detach()
            
            # Check for convergence
            if epoch > 0 and abs(loss.item() - best_loss) < 1e-6:
                print("Converged early at epoch", epoch)
                break
            
        # Initialize cluster centers using K-means
        kmeans = KMeans(n_clusters=self.model.n_clusters, n_init=20)
        z_np = best_z.cpu().numpy()
        y_pred = kmeans.fit_predict(z_np)
        y_pred = torch.tensor(y_pred, dtype=torch.long, device=self.device)
        self.model.cluster_layer.data = torch.tensor(
            kmeans.cluster_centers_, device=self.device
        )
        return y_pred
    
    def train(self, data, epochs, pretrain_epochs=100):
        print("Pretraining...")
        y_pred = self.pretrain(data, pretrain_epochs)
        
        print("Training...")
        for epoch in range(epochs):
            _, _, q = self.model(data.x, data.edge_index)
            
            if epoch % self.update_interval == 0:
                target_dist = self.calculate_target_dist(q.detach())
                max_probs, new_pred = q.max(1)
                new_pred = new_pred.to(self.device)
                y_pred = y_pred.to(self.device)
                
                delta_label = torch.sum(new_pred != y_pred).float() / new_pred.shape[0]
                delta_label = delta_label.item()
                
                y_pred = new_pred
                
                print(f'Epoch {epoch}: Label change = {delta_label:.4f}')
                
                # if delta_label < 0.001:
                #     print("Reached convergence threshold. Training stopped.")
                #     break
            
            loss, _ = self.train_step(data, target_dist)
            
            if (epoch + 1) % 10 == 0:
                print(f'Epoch {epoch+1}: Loss = {loss:.4f}')
        
        return y_pred
    

def prepare_data(edge_index, features, n_nodes, device):
    adj = torch.zeros((n_nodes, n_nodes), device=device)
    adj[edge_index[0], edge_index[1]] = 1
    
    class Data:
        def __init__(self):
            pass
    
    data = Data()
    data.x = features
    data.edge_index = edge_index
    data.adj = adj
    
    return data


# %%
class CDBNE_Evaluator:
    @staticmethod
    def cluster_acc(y_true, y_pred):
        """
        Calculate clustering accuracy using Hungarian algorithm
        """
        y_true = y_true.astype(np.int64)
        assert y_pred.size == y_true.size
        D = max(y_pred.max(), y_true.max()) + 1
        w = np.zeros((D, D), dtype=np.int64)
        
        # Count co-occurrence of assignments
        for i in range(y_pred.size):
            w[y_pred[i], y_true[i]] += 1
            
        # Find optimal one-to-one mapping
        ind = linear_sum_assignment(w.max() - w)
        ind = np.asarray(ind)
        ind = np.transpose(ind)
        
        # Return accuracy
        return sum([w[i, j] for i, j in ind]) * 1.0 / y_pred.size

    @staticmethod
    def evaluate(y_true, y_pred):
        """
        Evaluate clustering results using multiple metrics
        """
        # Convert to numpy arrays
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
        
        # Calculate metrics
        acc = CDBNE_Evaluator.cluster_acc(y_true, y_pred)
        nmi = normalized_mutual_info_score(y_true, y_pred)
        ari = adjusted_rand_score(y_true, y_pred)
        
        return {
            'ACC': acc,
            'NMI': nmi,
            'ARI': ari
        }


#%%
class CDBNE_Visualizer:
    @staticmethod
    def plot_tsne(embeddings, labels, title='t-SNE Visualization'):
        """
        Visualize embeddings using t-SNE
        """
        # Apply t-SNE
        tsne = TSNE(n_components=2, random_state=42)
        embeddings_2d = tsne.fit_transform(embeddings)
        
        # Plot
        plt.figure(figsize=(10, 8))
        scatter = plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], 
                            c=labels, cmap='tab10')
        plt.colorbar(scatter)
        plt.title(title)
        plt.show()


#%%
def main():
    torch.manual_seed(42)
    np.random.seed(42)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    dataset = Planetoid(root='/tmp/Cora', name='Cora', transform=T.NormalizeFeatures())
    data = dataset[0]
    
    data.x = data.x.to(device)
    data.edge_index = data.edge_index.to(device)
    
    # Model parameters
    input_dim = dataset.num_features
    hidden_dim = 256
    embedding_dim = 16
    n_clusters = dataset.num_classes
    alpha = 1.0  # degrees of freedom for Student's t-distribution
    
    model = CDBNE(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        embedding_dim=embedding_dim,
        n_clusters=n_clusters,
        alpha=alpha
    ).to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001, weight_decay=5e-3) # cora = 0.0001
    
    trainer = CDBNE_Trainer(
        model=model,
        optimizer=optimizer,
        device=device,
        beta=0.1,  # modularity loss weight
        gamma=1.0  # clustering loss weight
    )
    
    data_obj = prepare_data(data.edge_index, data.x, data.num_nodes, device)
    
    print("Starting training...")
    y_pred = trainer.train(data_obj, epochs=500, pretrain_epochs=100)
    
    y_pred_cpu = y_pred.cpu()
    metrics = CDBNE_Evaluator.evaluate(data.y.cpu().numpy(), y_pred_cpu.numpy())
    print("\nFinal Results:")
    for metric, value in metrics.items():
        print(f"{metric}: {value:.4f}")
    
    with torch.no_grad():
        embeddings, _, _ = model(data.x, data.edge_index)
        embeddings = embeddings.cpu().numpy()
    
    CDBNE_Visualizer.plot_tsne(embeddings, y_pred_cpu.numpy(), 'CDBNE Embeddings Visualization')


#%%
if __name__ == "__main__":
    main()


# %%
