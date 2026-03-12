import torch
class EarlyStopping:

    def __init__(self, patience=10, delta=0.0, path="best_model.pth"):

        self.patience = patience
        self.delta = delta
        self.path = path

        self.counter = 0
        self.best_score = None
        self.early_stop = False

    def __call__(self, score, model):

        if self.best_score is None:
            self.best_score = score
            self.save_model(model)

        elif score < self.best_score + self.delta:
            self.counter += 1
            print(f"EarlyStopping Counter: {self.counter}/{self.patience}")

            if self.counter >= self.patience:
                self.early_stop = True

        else:
            self.best_score = score
            self.save_model(model)
            self.counter = 0

    def save_model(self, model):

        torch.save(model.state_dict(), self.path)
        print("Best model saved.")