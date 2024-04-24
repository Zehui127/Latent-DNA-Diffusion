import os
import torch
from accelerate import Accelerator
from dna_encoder import *
from epd_loader import *

accelerator = Accelerator(cpu=True)

class TrainerConfig(LoaderConfig):
    # training parameters
    epoch: int = 1
    learning_rate: float = 1e-3
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    # model parameters
    num_layers: int = 5 # the indices of two to compress length (0.5^n)
    funnel_width: int = 64 # target width dimension

def main():
    config = TrainerConfig()
    # config.device = accelerator.device
    print("Training on", config.device)
    model = SequenceAE(num_layers=config.num_layers, funnel_width=config.funnel_width)
    model.to(config.device)

    print(list(model.parameters()))

    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    model, optimizer, train_data_loader, test_data_loader = accelerator.prepare(
        model, optimizer, train_loader, test_loader
    )

    print("=== Training...")
    for epoch in range(config.epoch):
        model = model.train()
        for batch_idx, data in enumerate(train_data_loader):
            data = data.to(config.device)
            y, x = model(data)

            loss = model.loss_function(y, x)
            accelerator.backward(loss=loss)
            optimizer.step()
            optimizer.zero_grad()

            if batch_idx % 100 == 0:
                print(f"Epoch: {epoch}, Loss: {loss.item():.4f}")
    print(">>> Training finished.")

    #  model validation
    print("=== Validating...")
    with torch.no_grad():
        loss_li = []
        for batch_idx, data in enumerate(test_data_loader): # use test_loader for now
            data = data.to(config.device)
            
            y, x = model(data)
            loss = model.loss_function(y, x)
            loss_li.append(loss)
        print(f"Validation Loss: {sum(loss_li)/len(loss_li):.4f}")

    #  model saving
    print(">>> Saving model...")
    torch.save(model.state_dict(), os.path.join(PATH, "model.pth"))

if __name__ == '__main__':
    main()