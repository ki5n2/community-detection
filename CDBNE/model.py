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
    def __init__(self, input_dim, hidden_dim, embedding_dim, n_clusters):
        """
        CDBNE: Community Detection Based on Network Embedding
        
        Args:
            input_dim: Number of input features
            hidden_dim: Dimension of hidden layer
            embedding_dim: Dimension of embedding
            n_clusters: Number of clusters/communities
        """
        super(CDBNE, self).__init__()
        
        # Graph Attention Auto-encoder
        # Encoder
        self.encoder_gat1 = GATConv(input_dim, hidden_dim)
        self.encoder_gat2 = GATConv(hidden_dim, embedding_dim)
        
        # Decoder
        self.decoder_gat1 = GATConv(embedding_dim, hidden_dim)
        self.decoder_gat2 = GATConv(hidden_dim, input_dim)
        
        # Clustering layer
        self.cluster_layer = nn.Parameter(torch.Tensor(n_clusters, embedding_dim))
        torch.nn.init.xavier_normal_(self.cluster_layer.data)
        
        # Parameters
        self.hidden_dim = hidden_dim
        self.embedding_dim = embedding_dim
        self.n_clusters = n_clusters
        
    def encode(self, x, edge_index):
        """
        Encode the input data using Graph Attention layers
        """
        # First GAT layer with ReLU activation
        h1 = F.relu(self.encoder_gat1(x, edge_index))
        # Second GAT layer
        z = self.encoder_gat2(h1, edge_index)
        return z
        
    def decode(self, z, edge_index):
        """
        Decode the embeddings back to reconstruct input
        """
        # First decoder GAT layer with ReLU activation
        h1 = F.relu(self.decoder_gat1(z, edge_index))
        # Second decoder GAT layer
        x_hat = self.decoder_gat2(h1, edge_index)
        return x_hat
        
    def forward(self, x, edge_index):
        """
        Forward pass through the network
        """
        # Get embeddings
        z = self.encode(x, edge_index)
        
        # Reconstruction
        x_hat = self.decode(z, edge_index)
        
        # Calculate cluster assignments (Q)
        q = self.get_cluster_prob(z)
        
        return z, x_hat, q
    
    def get_cluster_prob(self, z):
        """
        Calculate probability of assignments to clusters using Student's t-distribution
        """
        q = 1.0 / (1.0 + torch.sum(
            torch.pow(z.unsqueeze(1) - self.cluster_layer, 2), 2) / 1.0)
        q = q.pow(0.5)  # degree of freedom = 1
        q = (q.t() / torch.sum(q, 1)).t()
        return q

    def loss_reconstruction(self, x, x_hat, edge_index):
        """
        Calculate reconstruction loss
        """
        # Attribute reconstruction loss
        loss_attr = F.mse_loss(x_hat, x)
        
        # Structure reconstruction loss
        # Using inner product between node embeddings as edge predictions
        z = self.encode(x, edge_index)
        adj_pred = torch.sigmoid(torch.mm(z, z.t()))
        
        # Create adjacency matrix from edge_index
        adj_true = torch.zeros((x.size(0), x.size(0)), device=x.device)
        adj_true[edge_index[0], edge_index[1]] = 1
        
        loss_struct = F.binary_cross_entropy(adj_pred, adj_true)
        
        return loss_attr + loss_struct

    def loss_modularity(self, z, adj):
        """
        Calculate modularity loss
        """
        # Ensure adj is on the same device as the model
        adj = adj.to(z.device)
        
        # Calculate degrees
        degrees = torch.sum(adj, dim=1)
        m = torch.sum(adj) / 2
        
        # Calculate modularity matrix B
        B = adj - torch.outer(degrees, degrees) / (2 * m)
        
        # Calculate community assignments using softmax
        H = F.softmax(torch.mm(z, self.cluster_layer.t()), dim=1)
        
        # Calculate modularity
        Q = torch.trace(torch.mm(torch.mm(H.t(), B), H)) / (4 * m)
        
        return -Q  # Return negative since we want to maximize modularity


#%%
class CDBNE_Trainer:
    def __init__(self, model, optimizer, device, 
                 beta=0.1, gamma=1.0, update_interval=1):
        self.model = model
        self.optimizer = optimizer
        self.device = device
        self.beta = beta
        self.gamma = gamma
        self.update_interval = update_interval

    def calculate_target_dist(self, q):
        """
        Calculate auxiliary target distribution P
        Args:
            q: cluster soft assignments (size: n_samples x n_clusters)
        Returns:
            target distribution P
        """
        # Get weight of each sample to its assigned cluster
        weight = (q ** 2) / torch.sum(q, dim=0)
        # Normalize weight
        p = weight / torch.sum(weight, dim=1, keepdim=True)
        return p

    def train_step(self, data, target_dist=None):
        """
        Single training step
        """
        self.model.train()
        self.optimizer.zero_grad()
        
        # Forward pass
        z, x_hat, q = self.model(data.x, data.edge_index)
        
        # Calculate reconstruction and modularity losses
        loss_recon = self.model.loss_reconstruction(data.x, x_hat, data.edge_index)
        loss_mod = self.model.loss_modularity(z, data.adj)
        
        # Calculate clustering loss if target distribution is provided
        loss_cluster = 0
        if target_dist is not None:
            # KL divergence
            loss_cluster = F.kl_div(q.log(), target_dist)
        
        # Total loss
        loss = loss_recon - self.beta * loss_mod + self.gamma * loss_cluster
        
        # Backward pass
        loss.backward()
        self.optimizer.step()
        
        return loss.item(), q

    def pretrain(self, data, epochs):
        """
        Pretrain the model using only reconstruction and modularity loss
        """
        self.model.train()
        for epoch in range(epochs):
            self.optimizer.zero_grad()
            
            # Forward pass
            z, x_hat, _ = self.model(data.x, data.edge_index)
            
            # Calculate losses
            loss_recon = self.model.loss_reconstruction(data.x, x_hat, data.edge_index)
            loss_mod = self.model.loss_modularity(z, data.adj)
            
            # Total loss
            loss = loss_recon - self.beta * loss_mod
            
            # Backward pass
            loss.backward()
            self.optimizer.step()
            
            if (epoch + 1) % 100 == 0:
                print(f'Pretrain Epoch {epoch+1}: Loss = {loss.item():.4f}')
        
        # Initialize cluster centers using K-means
        kmeans = KMeans(n_clusters=self.model.n_clusters, n_init=20)
        z_np = z.detach().cpu().numpy()
        y_pred = kmeans.fit_predict(z_np)
        # Convert y_pred to tensor and move to correct device
        y_pred = torch.tensor(y_pred, dtype=torch.long, device=self.device)
        self.model.cluster_layer.data = torch.tensor(
            kmeans.cluster_centers_, device=self.device
        )
        
        return y_pred

    def train(self, data, epochs, pretrain_epochs=100):
        """
        Full training process
        """
        # Pretrain the model
        print("Pretraining...")
        y_pred = self.pretrain(data, pretrain_epochs)
        
        # Main training loop
        print("Training...")
        for epoch in range(epochs):
            # Get cluster assignments
            _, _, q = self.model(data.x, data.edge_index)
            
            # Update target distribution if needed
            if epoch % self.update_interval == 0:
                target_dist = self.calculate_target_dist(q.detach())
                
                # Monitor cluster assignment changes
                max_probs, new_pred = q.max(1)
                # Convert predictions to same device and type
                new_pred = new_pred.to(self.device)
                y_pred = y_pred.to(self.device)
                
                # Calculate label changes using torch operations
                delta_label = torch.sum(new_pred != y_pred).float() / new_pred.shape[0]
                delta_label = delta_label.item()
                
                y_pred = new_pred
                
                print(f'Epoch {epoch}: Label change = {delta_label:.4f}')
                
                # if delta_label < 0.001:  # Convergence check
                #     print("Reached convergence threshold. Training stopped.")
                #     break
            
            # Training step
            loss, _ = self.train_step(data, target_dist)
            
            if (epoch + 1) % 10 == 0:
                print(f'Epoch {epoch+1}: Loss = {loss:.4f}')
        
        return y_pred
    

def prepare_data(edge_index, features, n_nodes, device):
    """
    Prepare adjacency matrix and other data structures
    """
    # Create adjacency matrix
    adj = torch.zeros((n_nodes, n_nodes), device=device)  # Move to specified device
    adj[edge_index[0], edge_index[1]] = 1
    
    # Create data object
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

def main():
    # Set random seeds for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load dataset (using Cora as example)
    dataset = Planetoid(root='/tmp/Cora', name='Cora', transform=T.NormalizeFeatures())
    data = dataset[0]
    
    # Move data to device
    data.x = data.x.to(device)
    data.edge_index = data.edge_index.to(device)
    
    # Initialize model parameters
    input_dim = dataset.num_features
    hidden_dim = 256
    embedding_dim = 16
    n_clusters = dataset.num_classes
    
    # Create model
    model = CDBNE(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        embedding_dim=embedding_dim,
        n_clusters=n_clusters
    ).to(device)
    
    # Initialize optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    # Create trainer
    trainer = CDBNE_Trainer(
        model=model,
        optimizer=optimizer,
        device=device,
        beta=0.1,
        gamma=1.0
    )
    
    # Prepare data
    data_obj = prepare_data(
        data.edge_index,
        data.x,
        data.num_nodes,
        device  # Pass device to prepare_data
    )
    
    # Training
    print("Starting training...")
    y_pred = trainer.train(
        data=data_obj,
        epochs=500,
        pretrain_epochs=100
    )

    # Move predictions to CPU for evaluation
    y_pred_cpu = y_pred.cpu()
    
    # Evaluation
    metrics = CDBNE_Evaluator.evaluate(data.y.cpu().numpy(), y_pred_cpu.numpy())
    print("\nFinal Results:")
    for metric, value in metrics.items():
        print(f"{metric}: {value:.4f}")
    
    # Visualization
    with torch.no_grad():
        embeddings, _, _ = model(data.x, data.edge_index)
        embeddings = embeddings.cpu().numpy()
        
    CDBNE_Visualizer.plot_tsne(
        embeddings=embeddings,
        labels=y_pred_cpu.numpy(),
        title='CDBNE Embeddings Visualization'
    )

if __name__ == "__main__":
    main()
# %%
