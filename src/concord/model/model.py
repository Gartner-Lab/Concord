import torch
import torch.nn as nn
from .build_layer import get_normalization_layer, build_layers
from .. import logger

class ConcordModel(nn.Module):
    """
    A contrastive learning model for domain-aware and covariate-aware latent representations.

    This model consists of an encoder, decoder, and optional classifier head. 

    Attributes:
        domain_embedding_dim (int): Dimensionality of domain embeddings.
        input_dim (int): Input feature dimension.
        use_classifier (bool): Whether to include a classifier head.
        use_decoder (bool): Whether to include a decoder head.
        use_importance_mask (bool): Whether to include an importance mask for feature selection.
        encoder (nn.Sequential): Encoder layers.
        decoder (nn.Sequential, optional): Decoder layers.
        classifier (nn.Sequential, optional): Classifier head.
        importance_mask (nn.Parameter, optional): Learnable importance mask.
    """
    def __init__(self, input_dim, hidden_dim, num_domains, num_classes,  
                 domain_embedding_dim=0, 
                 covariate_embedding_dims={},
                 covariate_num_categories={},
                 encoder_dims=[], decoder_dims=[], 
                 dropout_prob: float = 0.0, 
                 norm_type='layer_norm', 
                 #encoder_append_cov=False, 
                 use_decoder=True, decoder_final_activation='leaky_relu',
                 use_classifier=False, use_importance_mask=False):
        """
        Initializes the Concord model.

        Args:
            input_dim (int): Number of input features.
            hidden_dim (int): Latent representation dimensionality.
            num_domains (int): Number of unique domains for embeddings.
            num_classes (int): Number of unique classes for classification.
            domain_embedding_dim (int, optional): Dimensionality of domain embeddings. Defaults to 0.
            covariate_embedding_dims (dict, optional): Dictionary mapping covariate keys to embedding dimensions.
            covariate_num_categories (dict, optional): Dictionary mapping covariate keys to category counts.
            encoder_dims (list, optional): List of encoder layer sizes. Defaults to empty list.
            decoder_dims (list, optional): List of decoder layer sizes. Defaults to empty list.
            dropout_prob (float, optional): Dropout probability for encoder/decoder layers. Defaults to 0.1.
            norm_type (str, optional): Normalization type ('layer_norm' or 'batch_norm'). Defaults to 'layer_norm'.
            use_decoder (bool, optional): Whether to include a decoder. Defaults to True.
            decoder_final_activation (str, optional): Activation function for decoder output. Defaults to 'leaky_relu'.
            use_classifier (bool, optional): Whether to include a classifier head. Defaults to False.
            use_importance_mask (bool, optional): Whether to learn an importance mask for input features. Defaults to False.
        """
        super().__init__()

        # Encoder
        self.domain_embedding_dim = domain_embedding_dim 
        self.input_dim = input_dim
        self.use_classifier = use_classifier
        self.use_decoder = use_decoder
        self.use_importance_mask = use_importance_mask

        total_embedding_dim = 0
        if domain_embedding_dim > 0:
            self.domain_embedding = nn.Embedding(num_embeddings=num_domains, embedding_dim=domain_embedding_dim)
            total_embedding_dim += domain_embedding_dim

        self.covariate_embeddings = nn.ModuleDict()
        for key, dim in covariate_embedding_dims.items():
            if dim > 0:
                self.covariate_embeddings[key] = nn.Embedding(num_embeddings=covariate_num_categories[key], embedding_dim=dim)
                total_embedding_dim += dim

        # self.encoder_append_cov = encoder_append_cov
        # if self.encoder_append_cov :
        #     encoder_input_dim = input_dim + total_embedding_dim
        # else:
        #     encoder_input_dim = input_dim

        encoder_input_dim = input_dim

        logger.info(f"Encoder input dim: {encoder_input_dim}")
        if self.use_decoder:
            decoder_input_dim = hidden_dim + total_embedding_dim
            logger.info(f"Decoder input dim: {decoder_input_dim}")
        if self.use_classifier:
            classifier_input_dim = hidden_dim # decoder_input_dim 
            logger.info(f"Classifier input dim: {classifier_input_dim}")

        # Encoder
        if encoder_dims:
            self.encoder = build_layers(encoder_input_dim, hidden_dim, encoder_dims, dropout_prob, norm_type)
        else:
            self.encoder = nn.Sequential(
                nn.Linear(encoder_input_dim, hidden_dim),
                get_normalization_layer(norm_type, hidden_dim),
                nn.LeakyReLU(0.1)
            )

        # Decoder
        if self.use_decoder:
            if decoder_dims:
                self.decoder = build_layers(decoder_input_dim, input_dim, decoder_dims, dropout_prob, norm_type, 
                                            final_layer_norm=False, final_layer_dropout=False, final_activation=decoder_final_activation)
            else:
                self.decoder = nn.Sequential(
                    nn.Linear(decoder_input_dim, input_dim)
                )

        # Classifier head
        if self.use_classifier:
            self.classifier = nn.Sequential(
                nn.Linear(classifier_input_dim, hidden_dim),
                get_normalization_layer(norm_type, hidden_dim),
                nn.LeakyReLU(0.1),
                nn.Linear(hidden_dim, num_classes)
            )

        self._initialize_weights()

        # Learnable mask for feature importance
        if self.use_importance_mask:
            self.importance_mask = nn.Parameter(torch.ones(input_dim))

    def forward(self, x, domain_labels=None, covariate_tensors=None):
        """
        Performs a forward pass through the model.

        Args:
            x (torch.Tensor): Input data.
            domain_labels (torch.Tensor, optional): Domain labels for embedding lookup.
            covariate_tensors (dict, optional): Dictionary of covariate labels.

        Returns:
            dict: A dictionary with encoded representations, decoded outputs (if enabled), 
                  classifier predictions (if enabled), and latent activations (if requested).
        """

        out = {}   

        out['encoded'] = self.encode(x)

        if self.use_decoder:
            x = out['encoded']
            embeddings = self.get_embeddings(domain_labels, covariate_tensors)
            if embeddings:
                x = torch.cat([x] + embeddings, dim=1)
            out['decoded'] = self.decoder(x)


        if self.use_classifier:
            x = out['encoded']
            # embeddings = self.get_embeddings(domain_labels, covariate_tensors)
            # if embeddings:
            #     x = torch.cat([x] + embeddings, dim=1)
            out['class_pred'] = self.classifier(x)

        return out

    def _initialize_weights(self):
        """Initializes model weights using Kaiming normal initialization."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def freeze_encoder(self):
        """Freezes encoder weights to prevent updates during training."""
        for param in self.encoder.parameters():
            param.requires_grad = False

    def load_model(self, path, device):
        """
        Loads a pre-trained model state.

        Args:
            path (str or Path): Path to the saved model checkpoint.
            device (torch.device): Device to load the model onto.
        """
        state_dict = torch.load(path, map_location=device)
        model_state_dict = self.state_dict()

        # Filter out layers with mismatched sizes
        filtered_state_dict = {k: v for k, v in state_dict.items() if
                               k in model_state_dict and v.size() == model_state_dict[k].size()}
        model_state_dict.update(filtered_state_dict)
        self.load_state_dict(model_state_dict, strict=False)

    def get_importance_weights(self):
        """
        Retrieves the learned importance weights for input features.

        Returns:
            torch.Tensor: The importance weights.
        """
        if self.use_importance_mask:
            #return torch.softmax(self.importance_mask, dim=0) * self.input_dim
            #return torch.relu(self.importance_mask)
            return torch.sigmoid(self.importance_mask)
        else:
            raise ValueError("Importance mask is not used in this model.")


    def encode(self, x):
        if self.use_importance_mask:
            x = x * self.get_importance_weights()

        return self.encoder(x)
    

    def get_embeddings(self, domain_labels=None, covariate_tensors=None):
        """
        Retrieves embeddings for the specified domain labels and covariate tensors.

        Args:
            domain_labels (torch.Tensor, optional): Domain labels for embedding lookup.
            covariate_tensors (dict, optional): Dictionary of covariate tensors.

        Returns:
            torch.Tensor: Concatenated embeddings.
        """
        embeddings = []
        if self.domain_embedding_dim > 0 and domain_labels is not None:
            domain_embeddings = self.domain_embedding(domain_labels)
            embeddings.append(domain_embeddings)

        if covariate_tensors is not None:
            for key, tensor in covariate_tensors.items():
                if key in self.covariate_embeddings:
                    embeddings.append(self.covariate_embeddings[key](tensor))

        if embeddings:
            return embeddings
        return None


    