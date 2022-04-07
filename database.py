import torch.nn.functional as F 

class ImageDatabase (Object):
    '''
       Database containing images indexed by latent vector.
    '''

    def __init__(self, dataset, encoder):
        '''
            dataset - tensor or list of examples ?
            encoder - ?
        '''
        self.dataset = dataset
        self.encoder = encoder

    def search(self, k, similarity=F.cosine_similarity):
        '''
            Search for k similar images according to user-provided similarity
            measure.
        '''
        pass
