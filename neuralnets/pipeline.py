import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
import lightning as L
from model import FunctionApproximator
import argparse
import PIL.Image as Image
import sympy as sp
import matplotlib.pyplot as plt

# ------Parsing------ #
parser = argparse.ArgumentParser(description='Train a neural network to approximate a function.')
parser.add_argument('--epochs', type=int, default=100, help='The number of epochs to train the neural network.')
parser.add_argument('--batch_size', type=int, default=64, help='The batch size to use when training the neural network.')
parser.add_argument('--learning_rate', type=float, default=1e-3, help='The learning rate to use when training the neural network.')
parser.add_argument('--function', type=str, default=None, help='The function to approximate. Must be in the form "y=f(x)".')
parser.add_argument('--image_path', type=str, default=None, help='The image to approximate. Must be a path to an image file.')
parser.add_argument('--image_size', type=int, default=64, help='The size of the image to approximate. (width x height)')
parser.add_argument('--rgb', action='store_true', help='Whether to use RGB values for the image approximation. If not set, the image will be converted to grayscale.')
args = parser.parse_args()

if args.function is None and args.image_path is None:
    raise ValueError('You must specify either a function to approximate or an image to approximate.')

if args.function is not None and args.image_path is not None:
    raise ValueError('You cannot specify both a function to approximate and an image to approximate.')    
# ------------------ #


# ------Config------ #

#default (normal y = f(x) is 1 input and 1 output):
input_size = 1
output_size = 1
is_image=False
image = None
function = None

if args.image_path is not None:
    is_image = True
    if args.rgb:
        input_size = 2
        output_size = 3
    else:
        input_size = 2
        output_size = 1
else:
    is_image = False  
# ----------------- #
    
    
# ----------------- #
def createDataset(data, is_image):
    if is_image == False:
        func = data
        xs = torch.linspace(-10, 10, 500).unsqueeze(1)
        ys = func(xs)
        dataset = TensorDataset(xs, ys)
    else:
        axis = torch.linspace(-1, 1, args.image_size)
        x_coord, y_coord = torch.meshgrid(axis, axis, indexing='ij')
        xs = torch.stack([x_coord, y_coord], dim=-1).reshape(-1, 2).float()
        ys = data.reshape(-1, 3).float() if args.rgb else data.reshape(-1, 1).float()
        dataset = TensorDataset(xs, ys)
        
    return dataset
    
# Create the dataset
if is_image:
    image = Image.open(args.image_path).resize((args.image_size, args.image_size))
    image = image.convert('RGB') if args.rgb else image.convert('L')
    image = torch.tensor(np.array(image) / 255.0).float()
    dataset = createDataset(image, is_image=True)

if not is_image:
    if not args.function.startswith('y='):
        raise ValueError('The function must be in the form "y=f(x)".')
    function = args.function.split('=')[1].strip().replace('^', '**')
    # parse the mathy function
    x = sp.symbols('x')
    parsed_function = sp.sympify(function)
    parsed_function = sp.lambdify(x, parsed_function, 'torch')
    dataset = createDataset(parsed_function, is_image=False)
# ----------------- #
    
    
loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
model = FunctionApproximator(input_size, output_size)

class LitModel(L.LightningModule):
    def __init__(self, model, learning_rate):
        super().__init__()
        self.model = model
        self.learning_rate = learning_rate

    def training_step(self, batch, batch_idx):
        x, y = batch
        output = self.model(x)
        loss = torch.nn.functional.mse_loss(output, y)
        self.log('train_loss', loss)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer
        
#---------------------#
class LiveFunctionVizCallback(L.Callback):
    def __init__(self, parsed_function):
        super().__init__()
        self.parsed_function = parsed_function
        plt.ion()
        self.fig, self.ax = plt.subplots()

    def on_train_epoch_end(self, trainer, pl_module):
        model = pl_module.model
        model.eval()

        with torch.no_grad():
            device = pl_module.device
            xs = torch.linspace(-10, 10, 500, device=device).unsqueeze(1)
            pred = model(xs).squeeze().detach().cpu().numpy()

            true_y = self.parsed_function(xs)
            if not torch.is_tensor(true_y):
                true_y = torch.full_like(xs, float(true_y))
            true_y = true_y.squeeze().cpu().numpy()

            x_np = xs.squeeze().cpu().numpy()

            self.ax.clear()
            self.ax.axhline(0, linewidth=1)
            self.ax.axvline(0, linewidth=1)
            self.ax.plot(x_np, true_y, label="True")
            self.ax.plot(x_np, pred, label="Prediction")
            self.ax.legend()
            self.ax.set_xlabel("x")
            self.ax.set_ylabel("y")
            self.ax.set_title(f"Epoch {trainer.current_epoch}")
            self.ax.grid(True, alpha=0.3)
            plt.pause(0.01)
            
class LiveImageVizCallback(L.Callback):
    def __init__(self, image_size, rgb):
        super().__init__()
        self.image_size = image_size
        self.rgb = rgb

        plt.ion()
        self.fig, self.ax = plt.subplots()

    def on_train_epoch_end(self, trainer, pl_module):
        model = pl_module.model
        model.eval()
        device = pl_module.device

        with torch.no_grad():
            axis = torch.linspace(-1, 1, self.image_size, device=device)
            x_coord, y_coord = torch.meshgrid(axis, axis, indexing='ij')
            xs = torch.stack([x_coord, y_coord], dim=-1).reshape(-1, 2).float()

            pred = model(xs).detach().cpu()

            self.ax.clear()

            if self.rgb:
                img = pred.reshape(self.image_size, self.image_size, 3).clamp(0, 1).numpy()
                self.ax.imshow(img)
            else:
                img = pred.reshape(self.image_size, self.image_size).clamp(0, 1).numpy()
                self.ax.imshow(img, cmap="gray")

            self.ax.set_title(f"Epoch {trainer.current_epoch}")
            self.ax.axis("off")
            plt.pause(0.01)
#---------------------#
        

lit_model = LitModel(model, args.learning_rate)
callbacks = []

if is_image:
    callbacks.append(LiveImageVizCallback(args.image_size, args.rgb))
else:
    callbacks.append(LiveFunctionVizCallback(parsed_function))

trainer = L.Trainer(max_epochs=args.epochs, callbacks=callbacks)
trainer.fit(lit_model, loader)