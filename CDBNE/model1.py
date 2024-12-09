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
from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score, homogeneity_score, completeness_score
from multiprocessing import Pool


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
    
    def loss_modularity(self, z, adj):
        """
        Calculate modularity loss with improved numerical stability
        """
        # Ensure input tensors are on correct device and type
        adj = adj.to(z.device).float()
        eps = 1e-10  # Small epsilon for numerical stability
        
        # Calculate degrees and normalize
        degrees = torch.sum(adj, dim=1)
        m = torch.sum(adj) / 2.0 + eps
        normalized_degrees = degrees / (2.0 * m)
        
        # Calculate modularity matrix with stable computation
        B = adj - torch.outer(normalized_degrees, normalized_degrees) * (2.0 * m)
        B = B / torch.max(torch.abs(B))  # Scale for numerical stability
        
        # Calculate community assignments using softmax with temperature
        temperature = 0.1
        H = F.softmax(torch.mm(z, self.cluster_layer.t()) / temperature, dim=1)
        
        # Calculate modularity Q with stable matrix operations
        HtB = torch.mm(H.t(), B)
        HtBH = torch.mm(HtB, H)
        Q = torch.trace(HtBH) / (4.0 * m)
        
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
    def __init__(self, model, optimizer, device, beta=0.1, gamma=1.0, 
                 patience=10, min_delta=1e-6, update_interval=10):
        self.model = model
        self.optimizer = optimizer
        self.device = device
        self.beta = beta
        self.gamma = gamma
        self.patience = patience
        self.min_delta = min_delta
        self.update_interval = update_interval
        
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=5, verbose=True, min_lr=1e-6
        )

    def calculate_target_dist(self, q):
        """
        Calculate target distribution P based on current soft assignments Q
        
        Args:
            q: Current soft assignment distributions (batch_size x n_clusters)
        Returns:
            Target distribution P
        """
        # Calculate weight of each cluster
        f = torch.sum(q, dim=0)  # 각 클러스터의 크기
        
        # Calculate auxiliary target distribution
        numerator = (q ** 2) / f
        denominator = torch.sum(numerator, dim=1, keepdim=True)
        
        # Normalize to get target distribution
        p = numerator / denominator
        
        return p

    def train(self, data, dataset_name, epochs, pretrain_epochs=100, validation_data=None):
        print("Pretraining...")
        y_pred = self.pretrain(data, pretrain_epochs)
        
        # 초기 시각화
        with torch.no_grad():
            embeddings, _, _ = self.model(data.x, data.edge_index)
            embeddings = embeddings.cpu().numpy()
            CDBNE_Visualizer.plot_tsne(embeddings, y_pred.cpu().numpy(), 
                                    'Initial Stage Visualization', f'/root/default/COMMUNITY_DETECTION/CDBNE/visual/{dataset_name}/initial_stage.png')
        
        print("Training...")
        best_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(epochs):
            # Forward pass
            z, x_hat, q = self.model(data.x, data.edge_index)
            
            # 중간 시각화 (epoch의 절반 지점)
            if epoch == epochs // 2:
                with torch.no_grad():
                    embeddings = z.cpu().numpy()
                    CDBNE_Visualizer.plot_tsne(embeddings, y_pred.cpu().numpy(), 
                                            'Mid-training Visualization', f'/root/default/COMMUNITY_DETECTION/CDBNE/visual/{dataset_name}/{epoch}_mid_training.png')
            
            # Update target distribution periodically
            if epoch % self.update_interval == 0:
                target_dist = self.calculate_target_dist(q.detach())
                max_probs, new_pred = q.max(1)
                
                # Monitor clustering stability
                delta_label = (new_pred != y_pred).float().mean().item()
                y_pred = new_pred
                
                print(f'Epoch {epoch}: Label change = {delta_label:.8f}')
            
            # Calculate loss
            loss, _ = self.train_step(data, target_dist)
            
            # Apply gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            # Learning rate scheduling
            self.scheduler.step(loss)
            
            # Early stopping
            # if loss < best_loss - self.min_delta:
            #     best_loss = loss
            #     patience_counter = 0
            # else:
            #     patience_counter += 1
            
            # if patience_counter >= self.patience:
            #     print(f"Early stopping at epoch {epoch}")
            #     break
            
            if (epoch + 1) % 10 == 0:
                print(f'Epoch {epoch+1}: Loss = {loss:.8f}')
                
                # Validate if validation data is provided
                if validation_data is not None:
                    val_loss = self.validate(validation_data)
                    print(f'Validation Loss = {val_loss:.8f}')
        
        # 최종 시각화
        with torch.no_grad():
            embeddings = z.cpu().numpy()
            CDBNE_Visualizer.plot_tsne(embeddings, y_pred.cpu().numpy(), 
                                    'Final Results Visualization', f'/root/default/COMMUNITY_DETECTION/CDBNE/visual/{dataset_name}/{epoch}_final_results.png')
        
        return y_pred
        
    def pretrain(self, data, epochs):
        """
        Pretrain the model to get initial embeddings
        """
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
            print((loss_recon / self.recon_scale))
            print(- self.beta * (loss_mod / self.mod_scale))
            # Backward pass with gradient clipping
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            
            # Learning rate scheduling
            self.scheduler.step(loss)
            
            # Print progress
            if (epoch + 1) % 10 == 0:
                print(f'Pretrain Epoch {epoch+1}: Loss = {loss.item():.8f} '
                    f'(Recon = {loss_recon.item():.8f}, Mod = {loss_mod.item():.8f})')
            
            # Save best embeddings
            if loss.item() < best_loss:
                best_loss = loss.item()
                best_z = z.detach()
            
            # Early stopping check
            # if epoch > 0 and abs(loss.item() - best_loss) < 1e-6:
            #     print(f"Converged early at epoch {epoch}")
            #     break
        
        # Initialize cluster centers using K-means
        kmeans = KMeans(n_clusters=self.model.n_clusters, n_init=20)
        z_np = best_z.cpu().numpy()
        y_pred = kmeans.fit_predict(z_np)
        y_pred = torch.tensor(y_pred, dtype=torch.long, device=self.device)
        
        # Update cluster layer with initial centroids
        self.model.cluster_layer.data = torch.tensor(
            kmeans.cluster_centers_, device=self.device
        )
        
        return y_pred

    def train_step(self, data, target_dist=None):
        """
        단일 training step을 수행하는 메소드
        
        Args:
            data: 학습 데이터
            target_dist: target distribution (optional)
        
        Returns:
            loss: 총 손실값
            q: 현재 클러스터 할당 확률
        """
        self.model.train()
        self.optimizer.zero_grad()
        
        # Forward pass
        z, x_hat, q = self.model(data.x, data.edge_index)
        
        # Calculate reconstruction loss
        loss_recon = self.model.loss_reconstruction(data.x, x_hat, data.edge_index)
        
        # Calculate modularity loss
        loss_mod = self.model.loss_modularity(z, data.adj)
        
        # Calculate clustering loss if target distribution is provided
        loss_cluster = 0
        if target_dist is not None:
            loss_cluster = torch.sum(target_dist * torch.log(target_dist / (q + 1e-7)))
        
        # Combine all losses
        loss = loss_recon - self.beta * loss_mod + self.gamma * loss_cluster
        
        # Backward pass
        loss.backward()
        
        # Update parameters
        self.optimizer.step()
        
        return loss.item(), q


#%%
def prepare_data(edge_index, features, n_nodes, device):
    """
    Prepare data with improved preprocessing and sparse matrix support
    """
    # Convert adjacency matrix to sparse format
    indices = edge_index
    values = torch.ones(indices.size(1))
    adj = torch.sparse_coo_tensor(
        indices, values, 
        size=(n_nodes, n_nodes),
        device=device
    )
    
    # Feature normalization
    features = F.normalize(features, p=2, dim=1)
    
    # Add self-loops to adjacency matrix
    adj = adj.to_dense()
    adj = adj + torch.eye(n_nodes, device=device)
    
    class Data:
        def __init__(self):
            self.x = features
            self.edge_index = edge_index
            self.adj = adj
            self.num_nodes = n_nodes
    
    return Data()


# %%
class CDBNE_Evaluator:
    @staticmethod
    def evaluate(y_true, y_pred):
        """
        Calculate multiple evaluation metrics
        """
        # Convert to numpy arrays
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
        
        # Calculate various metrics
        acc = CDBNE_Evaluator.cluster_acc(y_true, y_pred)
        nmi = normalized_mutual_info_score(y_true, y_pred)
        ari = adjusted_rand_score(y_true, y_pred)
        
        return {
            'ACC': acc,
            'NMI': nmi,
            'ARI': ari
        }

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


#%%
class CDBNE_Visualizer:
    @staticmethod
    def plot_tsne(embeddings, labels, title='t-SNE Visualization', save_path=None):
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
        
        if save_path:
            plt.savefig(save_path)
        plt.close()


#%%
def main():
    # 시드 설정
    torch.manual_seed(1)  # 1로 변경
    np.random.seed(1)     # 1로 변경
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # 데이터셋 로드
    dataset_name = 'Cora'
    dataset = Planetoid(root=f'/tmp/{dataset_name}', name=dataset_name, transform=T.NormalizeFeatures())
    data = dataset[0]
    data.x = data.x.to(device)
    data.edge_index = data.edge_index.to(device)
    
    # 모델 파라미터 설정
    input_dim = dataset.num_features
    hidden_dim = 256      # 원래대로 유지
    embedding_dim = 16    # 원래대로 유지
    n_clusters = dataset.num_classes
    alpha = 1.0          # 원래대로 유지
    
    # 모델 초기화
    model = CDBNE(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        embedding_dim=embedding_dim,
        n_clusters=n_clusters,
        alpha=alpha
    ).to(device)
    
    # 옵티마이저 설정
    optimizer = torch.optim.Adam(
        model.parameters(), 
        lr=0.0001,        # 원래대로 유지
        weight_decay=5e-3 # 원래대로 유지
    )
    
    # 트레이너 초기화
    trainer = CDBNE_Trainer(
        model=model,
        optimizer=optimizer,
        device=device,
        beta=0.1,         # modularity loss weight
        gamma=1.0,        # clustering loss weight
        patience=10,      # early stopping patience
        min_delta=1e-6,   # early stopping threshold
        update_interval=1  # target distribution 업데이트 간격
    )
    
    data_obj = prepare_data(data.edge_index, data.x, data.num_nodes, device)
    
    print("Starting training...")
    y_pred = trainer.train(data_obj, dataset_name, epochs=100, pretrain_epochs=100)  # pretrain epochs 수정
    
    y_pred_cpu = y_pred.cpu()
    metrics = CDBNE_Evaluator.evaluate(data.y.cpu().numpy(), y_pred_cpu.numpy())
    
    print("\nFinal Results:")
    for metric, value in metrics.items():
        print(f"{metric}: {value:.8f}")
    
    with torch.no_grad():
        embeddings, _, _ = model(data.x, data.edge_index)
        embeddings = embeddings.cpu().numpy()
        
    CDBNE_Visualizer.plot_tsne(embeddings, y_pred_cpu.numpy(), 'CDBNE Embeddings Visualization')


#%%
if __name__ == "__main__":
    main()
    
    
# %%
