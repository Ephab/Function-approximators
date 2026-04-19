import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
import lightning as L
from model import FunctionApproximator
import argparse
import PIL.Image as Image
import sympy as sp
import matplotlib.pyplot as plt

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


input_size = 1
output_size = 1
is_image=False
image = None
function = None
device = torch.device('mps' if torch.mps.is_available() else 'cpu')

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
    
    
loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)


# Training loop:
def train(steps=args.epochs, model:torch.nn.Module=None, optimizer='adam'):
    if model is None:
        raise ValueError('no model provided -.-')
    
    loss_fn = torch.nn.MSELoss()
    
    if optimizer == 'adam':
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
        
    elif optimizer == 'sgd':
        optimizer = torch.optim.SGD(model.parameters(), lr=args.learning_rate)
        
    elif optimizer == 'rmsprop':
        optimizer = torch.optim.RMSprop(model.parameters(), lr=args.learning_rate)
        
    else:
        raise ValueError(f"Unsupported optimizer: {optimizer}")
    
    # lr_sched = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=steps)
    lr_sched = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=10)
    
    for i in range(steps):
        epoch_loss = 0
        for X, y in loader:
            X, y = X.to(device), y.to(device)
            
            optimizer.zero_grad()
            output = model(X)
            
            loss = loss_fn(output, y)
            loss.backward()
            
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=20) #gradient clipping anything above 20
            
            epoch_loss += loss.item()
            optimizer.step()
            
        # lr_sched.step()
        lr_sched.step(epoch_loss / len(loader))
            
        yield {'epoch_loss': epoch_loss / len(loader),
               'learning_rate': optimizer.param_groups[0]['lr']}
            
def predict(model, dataset):
    all_x, all_y = dataset[:]
    all_x, all_y = all_x.to(device), all_y.to(device)
    
    model.eval()
    with torch.no_grad():
        return model(all_x)
        
model_adam = FunctionApproximator(input_size, output_size).to(device)
model_sgd = FunctionApproximator(input_size, output_size).to(device)
model_rmsprop = FunctionApproximator(input_size, output_size).to(device)

train_adam = train(steps=args.epochs, model=model_adam, optimizer='adam')
train_sgd = train(steps=args.epochs, model=model_sgd, optimizer='sgd')
train_rmsprop = train(steps=args.epochs, model=model_rmsprop, optimizer='rmsprop')

plt.ion()  # Turn on interactive mode
fig, axes = plt.subplots(1, 3, figsize=(15, 7))
titles = ['Adam', 'SGD', 'RMSprop']

print('training...')
for i in range(args.epochs):
    adam_output = next(train_adam)
    sgd_output = next(train_sgd)
    rmsprop_output = next(train_rmsprop)
    
    outputs = [predict(model_adam, dataset), predict(model_sgd, dataset), predict(model_rmsprop, dataset)]
    losses = [adam_output['epoch_loss'], sgd_output['epoch_loss'], rmsprop_output['epoch_loss']]
    learning_rates = [adam_output['learning_rate'], sgd_output['learning_rate'], rmsprop_output['learning_rate']]
    
    fig.suptitle(f'Epoch {i+1}/{args.epochs}')
    
    for j in range(len(outputs)):
        axes[j].clear()
        axes[j].set_title(f"{titles[j]} - Epoch Loss: {losses[j]:.4f}, LR: {learning_rates[j]:.5f}")
        
        if is_image:
            img_shape = (args.image_size, args.image_size, output_size)
            img_data = outputs[j].cpu().numpy().reshape(img_shape)
            img_data = np.clip(img_data, 0, 1)
            axes[j].imshow(img_data)
        else:
            all_x, all_y = dataset[:]
            axes[j].plot(all_x.numpy(), all_y.numpy(), label='Target')
            axes[j].plot(all_x.numpy(), outputs[j].cpu().numpy(), label='Prediction')
            axes[j].legend(loc='upper right')
            
    plt.pause(0.001)  
    
plt.ioff()
plt.show()


"""
To show the effect of momentum with adam
try the function: y= x^3 + 2*x^2 + x + 1
"""