import torch
import torch.nn.functional as F 

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def default_similarity(x1,x2):
    return F.cosine_similarity(x1, x2, dim=0)

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
        n = len(img.shape)
        img = torch.FloatTensor(img / 255).to(device)

        if n == 3:
            img = img.unsqueeze(0)
        elif n == 4:
            raise Exception('Invalid image')

        enc = self.encoder.forward(img)
        return enc.squeeze()

    def encode_images(self):
        db = []

        for label in self.dataset:
            for img in self.dataset[label]:
                db.append(self.encode_image(img))

        return db

    def search(self, img, k, similarity=default_similarity, threshold=1.0):
        '''
            Search for k similar images according to user-provided similarity
            measure.

            img - input image
        '''
        results = []
        n = 0
        enc = self.encode_image(img)

        for db_img in self.db:
            if n == k:
                return results
            enc_other = self.encode_image(db_img)
            if similarity(enc, enc_other) <= threshold:
                results.append(db_img)
                n += 1

        return results

if __name__ == '__main__':
    from base_model import ImageEncoder
    from dataset import load_cifar10

    train, test = load_cifar10()
    enc = ImageEncoder()
    db = ImageDatabase(test, enc)