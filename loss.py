import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Optional

def rotation_matrix_loss(R1, R2):
    # R1, R2: (..., 3, 3) shape
    R_delta = R1 @ R2.transpose(-1, -2)
    trace = R_delta[..., 0, 0] + R_delta[..., 1, 1] + R_delta[..., 2, 2]
    cos_theta = (trace - 1) / 2
    cos_theta = torch.clamp(cos_theta, -1.0, 1.0)
    theta = torch.acos(cos_theta)  # radians
    return theta.mean()  # or theta**2.mean() for MSE-like loss

def frobenius_rotation_loss(R1, R2):
    diff = R1 - R2
    # squared Frobenius norm
    loss = (diff ** 2).sum(dim=(-2, -1))  # = ||R1 - R2||_F^2
    return loss.mean()

def loss_refine(delta_t_pred, delta_t_gt, delta_R_pred, delta_R_gt, loss_cfg):
    
    w1=loss_cfg.trans.weight * (1/loss_cfg.trans_range if loss_cfg.trans_range> 0.0 else 1.0)
    w2=loss_cfg.rot.weight
    
    if loss_cfg.trans.get('type', None) == 'mse':
        loss_t = torch.nn.functional.mse_loss(delta_t_pred, delta_t_gt)
    else:
        raise NotImplementedError
    
    if loss_cfg.rot.get('type', None) == 'mse':
        loss_R = torch.nn.functional.mse_loss(delta_R_pred, delta_R_gt)
    elif loss_cfg.rot.get('type', None) == 'geo':
        loss_R = rotation_matrix_loss(delta_R_gt, delta_R_pred)
    elif loss_cfg.rot.get('type', None) == 'frob':
        loss_R = frobenius_rotation_loss(delta_R_gt, delta_R_pred)
    else:
        raise NotImplementedError
    # import pdb; pdb.set_trace()
    return w1 * loss_t + w2 * loss_R, loss_t, loss_R

class ContrastValidationLoss(nn.Module):
    """
    Pose-conditioned triplet loss for training the FoundationPose scorer network.

    This loss implements the contrast validation approach described in the FoundationPose paper,
    which uses triplet loss to train the pose ranking network by comparing positive and negative
    pose samples based on their ADD (Average Distance of Corresponding points) metric scores.
    """

    def __init__(self,
                 margin: float = 0.1,
                 rotation_threshold: float = 10.0,  # degrees
                 translation_threshold: float = 0.02,  # meters
                 add_threshold: float = 0.1):  # ADD threshold for positive samples
        """
        Args:
            margin: Contrastive margin (α in the paper)
            rotation_threshold: Rotation distance threshold for valid positive samples (degrees)
            translation_threshold: Translation distance threshold for valid positive samples (meters)
            add_threshold: ADD metric threshold for determining positive/negative samples
        """
        super().__init__()
        self.margin = margin
        self.rotation_threshold = np.radians(rotation_threshold)  # Convert to radians
        self.translation_threshold = translation_threshold
        self.add_threshold = add_threshold

    def geodesic_distance_rotation(self, R1: torch.Tensor, R2: torch.Tensor) -> torch.Tensor:
        """
        Compute geodesic distance between rotation matrices.

        Args:
            R1, R2: Rotation matrices of shape (B, 3, 3) or (3, 3)

        Returns:
            Geodesic distances in radians
        """
        if R1.dim() == 2:
            R1 = R1.unsqueeze(0)
        if R2.dim() == 2:
            R2 = R2.unsqueeze(0)

        # Compute relative rotation: R_rel = R2 * R1^T
        R_rel = torch.bmm(R2, R1.transpose(-1, -2))

        # Geodesic distance = arccos((trace(R_rel) - 1) / 2)
        trace = torch.diagonal(R_rel, dim1=-2, dim2=-1).sum(-1)
        # Clamp to avoid numerical issues with arccos
        cos_angle = torch.clamp((trace - 1) / 2, -1.0, 1.0)
        geodesic_dist = torch.acos(cos_angle)

        return geodesic_dist

    def translation_distance(self, t1: torch.Tensor, t2: torch.Tensor) -> torch.Tensor:
        """
        Compute Euclidean distance between translation vectors.

        Args:
            t1, t2: Translation vectors of shape (B, 3) or (3,)

        Returns:
            Euclidean distances
        """
        return torch.norm(t1 - t2, dim=-1)

    def compute_add_metric(self,
                           pred_poses: torch.Tensor,
                           gt_poses: torch.Tensor,
                           mesh_vertices: torch.Tensor) -> torch.Tensor:
        """
        Compute ADD (Average Distance of Corresponding points) metric.

        Args:
            pred_poses: Predicted poses (B, 4, 4) - homogeneous transformation matrices
            gt_poses: Ground truth poses (B, 4, 4)
            mesh_vertices: Mesh vertices (N, 3)

        Returns:
            ADD distances for each pose pair
        """
        B = pred_poses.shape[0]
        N = mesh_vertices.shape[0]

        # Extract rotation and translation from poses
        pred_R = pred_poses[:, :3, :3]  # (B, 3, 3)
        pred_t = pred_poses[:, :3, 3]  # (B, 3)
        gt_R = gt_poses[:, :3, :3]  # (B, 3, 3)
        gt_t = gt_poses[:, :3, 3]  # (B, 3)

        # Transform vertices with predicted poses
        vertices_expanded = mesh_vertices.unsqueeze(0).expand(B, -1, -1)  # (B, N, 3)
        pred_vertices = torch.bmm(vertices_expanded, pred_R.transpose(-1, -2)) + pred_t.unsqueeze(1)

        # Transform vertices with ground truth poses
        gt_vertices = torch.bmm(vertices_expanded, gt_R.transpose(-1, -2)) + gt_t.unsqueeze(1)

        # Compute ADD metric
        add_distances = torch.norm(pred_vertices - gt_vertices, dim=-1).mean(dim=-1)  # (B,)

        return add_distances

    def get_valid_pairs(self,
                        pred_poses: torch.Tensor,
                        gt_poses: torch.Tensor,
                        add_scores: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Determine valid positive and negative samples based on pose similarity and ADD scores.

        Args:
            pred_poses: Predicted poses (B, 4, 4)
            gt_poses: Ground truth poses (B, 4, 4)
            add_scores: ADD metric scores (B,)

        Returns:
            positive_mask: Boolean mask for valid positive samples
            negative_mask: Boolean mask for valid negative samples
        """
        B = pred_poses.shape[0]

        # Extract rotations and translations
        pred_R = pred_poses[:, :3, :3]
        pred_t = pred_poses[:, :3, 3]
        gt_R = gt_poses[:, :3, :3]
        gt_t = gt_poses[:, :3, 3]

        # Compute rotation and translation distances
        rot_distances = self.geodesic_distance_rotation(pred_R, gt_R)
        trans_distances = self.translation_distance(pred_t, gt_t)

        # Determine positive samples: close to ground truth in pose and good ADD score
        
        rot_close = (rot_distances < self.rotation_threshold)
        trans_close = (trans_distances < self.translation_threshold)
        pose_close = rot_close & trans_close
        add_good = add_scores < self.add_threshold
        # positive_mask = pose_close & add_good
        positive_mask = pose_close
        
        
        assert positive_mask.any()
        # import pdb; pdb.set_trace()
        # print(rot_close.float().mean(), trans_close.float().mean(), pose_close.float().mean())
        
        # Negative samples: all samples that are not positive
        negative_mask = ~positive_mask
        return positive_mask, negative_mask

    def forward(self,
                score_logits: torch.Tensor,
                pred_poses: torch.Tensor,
                gt_poses: torch.Tensor,
                mesh_vertices: torch.Tensor,
                pose_valid_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Compute the contrast validation loss.

        Args:
            score_logits: Score predictions from the network (B, L) where L is number of pose hypotheses
            pred_poses: Predicted poses (B, L, 4, 4)
            gt_poses: Ground truth poses (B, 4, 4)
            mesh_vertices: Object mesh vertices (N, 3)
            pose_valid_mask: Optional mask for valid poses (B, L)

        Returns:
            Contrast validation loss scalar
        """
        B, L = score_logits.shape
        device = score_logits.device

        # import pdb; pdb.set_trace()
        pred_poses = pred_poses.reshape(*score_logits.shape, 4, 4)
        # gt_poses = gt_poses.reshape(*score_logits.shape, 4, 4)
        # Expand ground truth poses to match prediction dimensions
        gt_poses_expanded = gt_poses.unsqueeze(1).expand(-1, L, -1, -1)  # (B, L, 4, 4)

        # Reshape for batch processing
        pred_poses_flat = pred_poses.contiguous().view(B * L, 4, 4)
        gt_poses_flat = gt_poses_expanded.contiguous().view(B * L, 4, 4)

        # Compute ADD scores
        add_scores = self.compute_add_metric(pred_poses_flat, gt_poses_flat, mesh_vertices)
        add_scores = add_scores.contiguous().view(B, L)

        total_loss = 0.0
        valid_pairs = 0

        # Process each batch sample
        for b in range(B):
            batch_scores = score_logits[b]  # (L,)
            batch_poses = pred_poses[b]  # (L, 4, 4)
            batch_gt = gt_poses[b]  # (4, 4)
            
            # add_scores = self.compute_add_metric(pred_poses[b], gt_poses_expanded[b], mesh_vertices)
            batch_add = add_scores[b]  # (L,)

            # Get ground truth pose in the right format for comparison
            gt_expanded = batch_gt.unsqueeze(0).expand(L, -1, -1)  # (L, 4, 4)

            # Get valid positive and negative samples
            pos_mask, neg_mask = self.get_valid_pairs(batch_poses, gt_expanded, batch_add)

            if pose_valid_mask is not None:
                # Apply additional validity mask
                batch_valid = pose_valid_mask[b]
                pos_mask = pos_mask & batch_valid
                neg_mask = neg_mask & batch_valid

            # Get indices of positive and negative samples
            pos_indices = torch.where(pos_mask)[0]
            neg_indices = torch.where(neg_mask)[0]

            if len(pos_indices) == 0 or len(neg_indices) == 0:
                raise NotImplementedError
                continue

            # Compute pairwise triplet loss
            batch_loss = 0.0
            batch_pairs = 0

            
            for pos_idx in pos_indices:
                for neg_idx in neg_indices:
                    if pos_idx != neg_idx:
                        # Triplet loss: max(S(i-) - S(i+) + α, 0)
                        # loss_val = torch.clamp(batch_scores[neg_idx] - batch_scores[pos_idx] + self.margin,min=0.0)
                        loss_val = F.softplus(batch_scores[neg_idx] - batch_scores[pos_idx] + self.margin)
                        batch_loss += loss_val
                        batch_pairs += 1

            if batch_pairs > 0:
                total_loss += batch_loss / batch_pairs
                valid_pairs += 1

        if valid_pairs > 0:
            return total_loss / valid_pairs
        else:
            # Return small loss if no valid pairs found
            # import pdb; pdb.set_trace()
            return False


# Example usage and training function
def train_scorer_with_contrast_loss(model: nn.Module,
                                    dataloader,
                                    optimizer,
                                    device: str = 'cuda',
                                    num_epochs: int = 50):
    """
    Training loop example using the contrast validation loss.

    Args:
        model: ScoreNetMultiPair model
        dataloader: DataLoader providing training batches
        optimizer: Optimizer (e.g., Adam)
        device: Training device
        num_epochs: Number of training epochs
    """
    model.train()
    contrast_loss = ContrastValidationLoss().to(device)

    for epoch in range(num_epochs):
        epoch_loss = 0.0
        num_batches = 0

        for batch_idx, batch_data in enumerate(dataloader):
            # Extract batch data (this depends on your dataset structure)
            rgb_A = batch_data['rgbAs'].to(device)  # (B*L, C, H, W)
            rgb_B = batch_data['rgbBs'].to(device)  # (B*L, C, H, W)
            xyz_A = batch_data['xyz_mapAs'].to(device)  # (B*L, 3, H, W)
            xyz_B = batch_data['xyz_mapBs'].to(device)  # (B*L, 3, H, W)

            pred_poses = batch_data['pred_poses'].to(device)  # (B, L, 4, 4)
            gt_poses = batch_data['gt_poses'].to(device)  # (B, 4, 4)
            mesh_vertices = batch_data['mesh_vertices'].to(device)  # (N, 3)

            B, L = pred_poses.shape[:2]

            # Prepare input tensors
            A = torch.cat([rgb_A, xyz_A], dim=1)  # Concatenate RGB and XYZ
            B = torch.cat([rgb_B, xyz_B], dim=1)

            optimizer.zero_grad()

            # Forward pass
            output = model(A, B, L)
            score_logits = output['score_logit']  # (B, L)

            # Compute contrast validation loss
            loss = contrast_loss(score_logits, pred_poses, gt_poses, mesh_vertices)

            # Backward pass
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            num_batches += 1

            if batch_idx % 100 == 0:
                print(f'Epoch {epoch}, Batch {batch_idx}, Loss: {loss.item():.4f}')

        avg_loss = epoch_loss / num_batches
        print(f'Epoch {epoch} completed. Average Loss: {avg_loss:.4f}')

# Usage example:
# criterion = ContrastValidationLoss(margin=0.1, rotation_threshold=10.0)
# train_scorer_with_contrast_loss(model, train_dataloader, optimizer)