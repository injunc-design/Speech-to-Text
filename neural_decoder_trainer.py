import os
import pickle
import time

from edit_distance import SequenceMatcher
import hydra
import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader
import torch.nn.functional as F

from .model import GRUDecoder, grad_reverse
from .dataset import SpeechDataset

import torch.optim.lr_scheduler

# --- PATCH START: Fix for PyTorch < 2.0 ---
if not hasattr(torch.optim.lr_scheduler, "LRScheduler"):
    # Create the alias 'LRScheduler' pointing to the hidden '_LRScheduler'
    torch.optim.lr_scheduler.LRScheduler = torch.optim.lr_scheduler._LRScheduler
# --- PATCH END ---

# Now you can safely import from the library
from pytorch_optimizer import Lookahead

def get_smart_sgd_lr(adam_optimizer, beta2=0.999):
    """
    Calculates the global SGD learning rate by projecting the 
    adaptive learning rates of Adam onto a single scalar.
    """
    print("[SWATS] Calculating projection from Adam state...")
    
    # 1. Get the current step (needed for bias correction)
    step = 0
    for group in adam_optimizer.param_groups:
        for p in group['params']:
            if p.grad is None: continue
            state = adam_optimizer.state[p]
            if 'step' in state:
                step = state['step']
                if isinstance(step, torch.Tensor): 
                    step = step.item()
                break
        if step > 0: break
    
    if step == 0:
        print("[SWATS] Warning: Optimizer has no step count. Defaulting to 0.01")
        return 0.01

    # 2. Calculate Bias Correction for the Second Moment (v_t)
    bias_correction = 1 - beta2 ** step
    
    effective_lrs = []
    
    # 3. Iterate over all parameters to find mean effective step size
    for group in adam_optimizer.param_groups:
        base_lr = group['lr']
        eps = group['eps'] # Use the epsilon defined in the optimizer
        
        for p in group['params']:
            if p.grad is None: continue
            
            state = adam_optimizer.state[p]
            if 'exp_avg_sq' not in state: continue 
            
            # Get the variance vector (v)
            v = state['exp_avg_sq']
            
            # Correct bias: v_hat = v / (1 - beta2^t)
            v_hat = v / bias_correction
            
            # Calculate Effective LR vector: base_lr / (sqrt(v_hat) + eps)
            # We assume the step direction is roughly consistent
            layer_effective_lr = base_lr / (torch.sqrt(v_hat) + eps)
            
            # We take the mean of this tensor to get a scalar for this parameter
            effective_lrs.append(layer_effective_lr.mean().item())

    if len(effective_lrs) == 0:
        return 0.01 
        
    # 4. Project to a single scalar (Arithmetic Mean of all effective LRs)
    projected_lr = np.mean(effective_lrs)
    return projected_lr


def getDatasetLoaders(
    datasetName,
    batchSize,
):
    with open(datasetName, "rb") as handle:
        loadedData = pickle.load(handle)

    def _padding(batch):
        X, y, X_lens, y_lens, days = zip(*batch)
        X_padded = pad_sequence(X, batch_first=True, padding_value=0)
        y_padded = pad_sequence(y, batch_first=True, padding_value=0)

        return (
            X_padded,
            y_padded,
            torch.stack(X_lens),
            torch.stack(y_lens),
            torch.stack(days),
        )

    train_ds = SpeechDataset(loadedData["train"], transform=None)
    test_ds = SpeechDataset(loadedData["test"])

    train_loader = DataLoader(
        train_ds,
        batch_size=batchSize,
        shuffle=True,
        num_workers=0,
        pin_memory=True,
        collate_fn=_padding,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=batchSize,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
        collate_fn=_padding,
    )

    return train_loader, test_loader, loadedData

def trainModel(args):
    os.makedirs(args["outputDir"], exist_ok=True)
    torch.manual_seed(args["seed"])
    np.random.seed(args["seed"])
    device = "cuda"

    with open(args["outputDir"] + "/args", "wb") as file:
        pickle.dump(args, file)

    trainLoader, testLoader, loadedData = getDatasetLoaders(
        args["datasetPath"],
        args["batchSize"],
    )

    model = GRUDecoder(
        neural_dim=args["nInputFeatures"],
        n_classes=args["nClasses"],
        hidden_dim=args["nUnits"],
        layer_dim=args["nLayers"],
        nDays=len(loadedData["train"]),
        dropout=args["dropout"],
        device=device,
        strideLen=args["strideLen"],
        kernelLen=args["kernelLen"],
        gaussianSmoothWidth=args["gaussianSmoothWidth"],
        bidirectional=args["bidirectional"],
    ).to(device)

    loss_ctc = torch.nn.CTCLoss(blank=0, reduction="mean", zero_infinity=True)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args["lrStart"],
        betas=(0.9, 0.999),
        eps=0.1,
        weight_decay=args["l2_decay"],
    )
    
    # optimizer = torch.optim.SGD(
    # model.parameters(),
    # lr=args["lrStart"], # You might need to TUNE this (often 0.1 or 0.01 for SGD)
    # momentum=0.9,
    # weight_decay=args["l2_decay"],
    # nesterov=True
    # )
    

    # optimizer = MADGRAD(
    #     model.parameters(),
    #     lr=args["lrStart"],     # Usually works with similar LR to Adam
    #     momentum=0.9,
    #     weight_decay=args["l2_decay"],
    #     eps=1e-3                # Uses a naturally smaller eps
    # )
    
    # Define your base SGD
    # base_optimizer = torch.optim.SGD(
    #     model.parameters(), 
    #     lr=args["lrStart"], 
    #     momentum=0.9
    # )

    # # Wrap it with Lookahead
    # optimizer = Lookahead(
    #     base_optimizer,
    #     k=5,   # Look ahead 5 steps
    #     alpha=0.8 # Merge explicitly (0.5 means average fast and slow)
    # )
    
    
    scheduler = torch.optim.lr_scheduler.LinearLR(
        optimizer,
        start_factor=1.0,
        end_factor=args["lrEnd"] / args["lrStart"],
        total_iters=args["nBatch"],
    )

    # # NEW define parameters
    # SWITCH_BATCH = args["nBatch"] * 0.8  # Example: Switch halfway through total batches
    # SGD_LR_START = args["lrStart"] / 2 # Start SGD with a lower LR
    # SGD_MOMENTUM = 0.9                   # Momentum is critical for SGD
    # SGD_LR_END = args["lrEnd"] / 2  # New, lower final LR for fine-tuning
    
    # scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
    #     optimizer,
    #     T_0=args["nBatch"] // 3, # Restart 3 times during training
    #     T_mult=2,                # Each cycle gets 2x longer
    #     eta_min=args["lrStart"] * 0.01 # Don't go all the way to 0
    # )

    
    # --- SWATS CONFIGURATION ---
    # Switch at 40% or 50% of batches usually works best for projection
    SWITCH_BATCH = int(args["nBatch"] * 0.4) 
    switched_to_sgd = False
    
    # --train--
    testLoss = []
    testCER = []
    startTime = time.time()
    for batch in range(args["nBatch"]):
 
#         The Adam-to-SGD Switch Logic
#         if batch == SWITCH_BATCH:
#             print("-" * 50)
#             print(f"Batch {batch}: Switching from Adam to SGD!")
            
#             # 1. Re-initialize the Optimizer (SGD)
#             optimizer = torch.optim.SGD(
#                 model.parameters(),
#                 lr=SGD_LR_START,
#                 momentum=SGD_MOMENTUM,
#                 weight_decay=args["l2_decay"], # Keep L2 decay
#             )
            
#             # 2. Re-initialize the Scheduler for the SGD phase
#             remaining_iters = args["nBatch"] - batch
            
#             scheduler = torch.optim.lr_scheduler.LinearLR(
#                 optimizer,
#                 start_factor=1.0,
#                 end_factor=SGD_LR_END / SGD_LR_START,
#                 total_iters=remaining_iters,
#             )
#             print(f"New Optimizer: SGD (LR={SGD_LR_START}, Momentum={SGD_MOMENTUM})")
#             print(f"New Scheduler: LinearLR over {remaining_iters} batches.")
#             print("-" * 50)
        
        if batch == SWITCH_BATCH and not switched_to_sgd:
            print("-" * 50)
            print(f"Batch {batch}: Initiating SWATS (Smart Switch to SGD)...")
            
            # A. Calculate the projected Learning Rate
            # We pass 0.999 as beta2 because that matches your Adam init above
            smart_sgd_lr = get_smart_sgd_lr(optimizer, beta2=0.999)
            
            print(f"Adam LR was: {args['lrStart']}")
            print(f"Projected SGD LR is: {smart_sgd_lr:.6f}")
            
            # B. Re-initialize Optimizer as SGD
            optimizer = torch.optim.SGD(
                model.parameters(),
                lr=smart_sgd_lr,
                momentum=0.9, # Momentum is critical for SGD
                weight_decay=args["l2_decay"], 
            )
            
            # C. Re-initialize Scheduler
            # We need a new scheduler that decays from the NEW lr to the FINAL lr
            # over the remaining batches.
            remaining_iters = args["nBatch"] - batch
            
            # Calculate what the end factor should be relative to the new SGD start
            # If we want to end at args["lrEnd"], the factor is lrEnd / smart_sgd_lr
            new_end_factor = args["lrEnd"] / smart_sgd_lr
            
            scheduler = torch.optim.lr_scheduler.LinearLR(
                optimizer,
                start_factor=1.0,
                end_factor=new_end_factor,
                total_iters=remaining_iters,
            )
            
            switched_to_sgd = True
            print(f"Switched to SGD with LR={smart_sgd_lr:.5f}, Momentum=0.9")
            print("-" * 50)
            
        model.train()

        X, y, X_len, y_len, dayIdx = next(iter(trainLoader))
        X, y, X_len, y_len, dayIdx = (
            X.to(device),
            y.to(device),
            X_len.to(device),
            y_len.to(device),
            dayIdx.to(device),
        )

        # Noise augmentation is faster on GPU
        if args["whiteNoiseSD"] > 0:
            X += torch.randn(X.shape, device=device) * args["whiteNoiseSD"]

        if args["constantOffsetSD"] > 0:
            X += (
                torch.randn([X.shape[0], 1, X.shape[2]], device=device)
                * args["constantOffsetSD"]
            )

        # Compute prediction error
#         pred = model.forward(X, dayIdx)

#         loss = loss_ctc(
#             torch.permute(pred.log_softmax(2), [1, 0, 2]),
#             y,
#             ((X_len - model.kernelLen) / model.strideLen).to(torch.int32),
#             y_len,
#         )
#         loss = torch.sum(loss)

        pred, day_logits, hid = model.forward(X, dayIdx)
    
        ctc_loss = loss_ctc(
            torch.permute(pred.log_softmax(2), [1, 0, 2]),
            y,
            ((X_len - model.kernelLen) / model.strideLen).to(torch.int32),
            y_len,
        )
        ctc_loss = torch.sum(ctc_loss)

        # Day classification loss (adversarial)
        B, T, feat_dim = hid.shape
        rev_features = grad_reverse(hid, lambd=1.0)
        day_logits = model.day_classifier(rev_features.reshape(-1, feat_dim))
        day_targets = dayIdx.unsqueeze(1).repeat(1, T).reshape(-1)
        day_loss = F.cross_entropy(day_logits, day_targets)

        # Combined loss
        alpha = 0.1
        loss = ctc_loss + alpha * day_loss

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()

        # print(endTime - startTime)

        # Eval
        if batch % 100 == 0:
            with torch.no_grad():
                model.eval()
                allLoss = []
                total_edit_distance = 0
                total_seq_length = 0
                for X, y, X_len, y_len, testDayIdx in testLoader:
                    X, y, X_len, y_len, testDayIdx = (
                        X.to(device),
                        y.to(device),
                        X_len.to(device),
                        y_len.to(device),
                        testDayIdx.to(device),
                    )

                    # pred = model.forward(X, testDayIdx)
                    pred, day_logits, hid = model.forward(X, testDayIdx)
                    loss = loss_ctc(
                        torch.permute(pred.log_softmax(2), [1, 0, 2]),
                        y,
                        ((X_len - model.kernelLen) / model.strideLen).to(torch.int32),
                        y_len,
                    )
                    loss = torch.sum(loss)
                    allLoss.append(loss.cpu().detach().numpy())

                    adjustedLens = ((X_len - model.kernelLen) / model.strideLen).to(
                        torch.int32
                    )
                    for iterIdx in range(pred.shape[0]):
                        decodedSeq = torch.argmax(
                            torch.tensor(pred[iterIdx, 0 : adjustedLens[iterIdx], :]),
                            dim=-1,
                        )  # [num_seq,]
                        decodedSeq = torch.unique_consecutive(decodedSeq, dim=-1)
                        decodedSeq = decodedSeq.cpu().detach().numpy()
                        decodedSeq = np.array([i for i in decodedSeq if i != 0])

                        trueSeq = np.array(
                            y[iterIdx][0 : y_len[iterIdx]].cpu().detach()
                        )

                        matcher = SequenceMatcher(
                            a=trueSeq.tolist(), b=decodedSeq.tolist()
                        )
                        total_edit_distance += matcher.distance()
                        total_seq_length += len(trueSeq)

                avgDayLoss = np.sum(allLoss) / len(testLoader)
                cer = total_edit_distance / total_seq_length

                endTime = time.time()
                print(
                    f"batch {batch}, ctc loss: {avgDayLoss:>7f}, cer: {cer:>7f}, time/batch: {(endTime - startTime)/100:>7.3f}"
                )
                startTime = time.time()

            if len(testCER) > 0 and cer < np.min(testCER):
                torch.save(model.state_dict(), args["outputDir"] + "/modelWeights")
            testLoss.append(avgDayLoss)
            testCER.append(cer)

            tStats = {}
            tStats["testLoss"] = np.array(testLoss)
            tStats["testCER"] = np.array(testCER)

            with open(args["outputDir"] + "/trainingStats", "wb") as file:
                pickle.dump(tStats, file)


def loadModel(modelDir, nInputLayers=24, device="cuda"):
    modelWeightPath = modelDir + "/modelWeights"
    with open(modelDir + "/args", "rb") as handle:
        args = pickle.load(handle)

    model = GRUDecoder(
        neural_dim=args["nInputFeatures"],
        n_classes=args["nClasses"],
        hidden_dim=args["nUnits"],
        layer_dim=args["nLayers"],
        nDays=nInputLayers,
        dropout=args["dropout"],
        device=device,
        strideLen=args["strideLen"],
        kernelLen=args["kernelLen"],
        gaussianSmoothWidth=args["gaussianSmoothWidth"],
        bidirectional=args["bidirectional"],
    ).to(device)

    model.load_state_dict(torch.load(modelWeightPath, map_location=device))
    return model


@hydra.main(version_base="1.1", config_path="conf", config_name="config")
def main(cfg):
    cfg.outputDir = os.getcwd()
    trainModel(cfg)

if __name__ == "__main__":
    main()