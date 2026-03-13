import torch
from thop import profile
from model import LightSegNet_V3

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def compute_flops_from_saved(model_path):

    model = LightSegNet_V3().to(DEVICE)

    model.load_state_dict(torch.load(model_path, map_location=DEVICE))

    model.eval()

    dummy_input = torch.randn(1,3,300,300).to(DEVICE)

    flops, params = profile(model, inputs=(dummy_input,), verbose=False)

    flops_giga = flops / 1e9

    return flops_giga, params

def compute_flops(model,device):

    model=model.to(device)

    model.eval()

    dummy_input = torch.randn(1,3,300,300).to(DEVICE)

    flops, params = profile(model, inputs=(dummy_input,), verbose=False)

    flops_giga = flops / 1e9

    return flops_giga, params





def ranking_score(dice_score, flops_giga):

    return dice_score / flops_giga

def read_dice_score(file_path="score.txt"):

    with open(file_path, "r") as f:
        avg_dice = float(f.read().strip())

    return avg_dice

if __name__ == "__main__":

    flopsgega, params = compute_flops_from_saved("model.pth")

    avg_dice = read_dice_score()

    ranking_score = avg_dice / flopsgega

    print("Ranking Score:", ranking_score)

    print(f"FLOPs: {flopsgega:.3f} GFLOPs")
    print(f"Params: {params/1e6:.2f} Million")
    print(f"Score: {ranking_score}")