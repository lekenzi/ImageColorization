import torch
from tqdm.auto import tqdm
from image_colorizer import device
from image_colorizer.helper import visualize
from image_colorizer.model import MainModel
from image_colorizer.model.init_weights import init_model
from image_colorizer.dataset import ColorizationDataLoader

class ColorizationTrainer:
    """
    Class to handle the training process of an image colorization model.
    """

    def __init__(self, model, train_dl, val_dl, epochs, display_every=40, save_every=5, save_path='model_checkpoint.pth'):
        self.model = model
        self.train_dl = train_dl
        self.val_dl = val_dl
        self.epochs = epochs
        self.display_every = display_every
        self.save_every = save_every
        self.save_path = save_path
        self.loss_meter_dict = self.create_loss_meters()

    def create_loss_meters(self):
        """
        Initialize loss meters for logging purposes.
        """
        return {'loss': 0}

    def update_losses(self, model, loss_meter_dict, count):
        """
        Update loss meters with the current losses.
        """
        loss_meter_dict['loss'] += model.loss * count

    def log_results(self, loss_meter_dict):
        """
        Log the current results.
        """
        print(f"Loss: {loss_meter_dict['loss'] / len(self.train_dl.dataset)}")


    def train(self):
        """
        Train the model.
        """
        data = next(iter(self.val_dl))  # getting a batch for visualizing the model output after fixed intervals
        for e in range(self.epochs):
            self.loss_meter_dict = self.create_loss_meters()  # resetting the loss meters
            i = 0
            for data in tqdm(self.train_dl):
                self.model.setup_input(data)
                self.model.optimize()
                self.update_losses(self.model, self.loss_meter_dict, count=data['L'].size(0))
                i += 1
                if i % self.display_every == 0:
                    print(f"\nEpoch {e+1}/{self.epochs}")
                    print(f"Iteration {i}/{len(self.train_dl)}")
                    self.log_results(self.loss_meter_dict)
                    visualize(self.model, data)
                if i % self.save_every == 0:
                    torch.save(self.model.state_dict(), self.save_path)
                    print(f"Model saved at iteration {i} of epoch {e+1}")

model = MainModel()
model = init_model(model, device)

colorizationDataLoader = ColorizationDataLoader()
train_dl, val_dl = colorizationDataLoader.get_dataloaders()

trainer = ColorizationTrainer(model, train_dl, val_dl, epochs=10)


trainer.train()

# Path: image_colorizer/model.py