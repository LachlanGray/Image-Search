import torch
import torch.nn.functional as F 

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class ImageDatabase (object):
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
        self.db = self.encode_images()

    def encode_image(self, img):
        img = torch.FloatTensor(img / 255).to(device)
        # img = torch.from_numpy(img / 255).to(device)
        img = img.unsqueeze(0)
        # raise Exception('{}'.format(img))
        enc = self.encoder.forward(img)
        return enc.squeeze()

    def encode_images(self):
        db = []

        for label in self.dataset:
            for img in self.dataset[label]:
                db.append(self.encode_image(img))

        return db

    def search(self, img, k, similarity=F.cosine_similarity):
        '''
            Search for k similar images according to user-provided similarity
            measure.

            img - input image
        '''
        pass

if __name__ == '__main__':
    from base_model import ImageEncoder
    from dataset import load_cifar10

    train, test = load_cifar10()
    enc = ImageEncoder()
    db = ImageDatabase(test, enc)