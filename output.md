This file is a merged representation of the entire codebase, combined into a single document by Repomix.

<file_summary>
This section contains a summary of this file.

<purpose>
This file contains a packed representation of the entire repository's contents.
It is designed to be easily consumable by AI systems for analysis, code review,
or other automated processes.
</purpose>

<file_format>
The content is organized as follows:
1. This summary section
2. Repository information
3. Directory structure
4. Repository files (if enabled)
5. Multiple file entries, each consisting of:
  - File path as an attribute
  - Full contents of the file
</file_format>

<usage_guidelines>
- This file should be treated as read-only. Any changes should be made to the
  original repository files, not this packed version.
- When processing this file, use the file path to distinguish
  between different files in the repository.
- Be aware that this file may contain sensitive information. Handle it with
  the same level of security as you would the original repository.
</usage_guidelines>

<notes>
- Some files may have been excluded based on .gitignore rules and Repomix's configuration
- Binary files are not included in this packed representation. Please refer to the Repository Structure section for a complete list of file paths, including binary files
- Files matching patterns in .gitignore are excluded
- Files matching default ignore patterns are excluded
- Files are sorted by Git change count (files with more changes are at the bottom)
</notes>

</file_summary>

<directory_structure>
attacks/
  l0_attack/
    __init__.py
    L0_Linf_PGD.py
    L0_PGD.py
    L0_Sigma_PGD.py
    L0_Utils.py
    utils.py
  l1_attack/
    __init__.py
    autoattack.py
    autopgd_base.py
    checks.py
    other_utils.py
    output.md
    state.py
  l2_attack/
    autopgd_base.py
    checks.py
    DataManagerPytorch.py
    other_utils.py
    repo.md
  linf_attack/
    APGD_Linf.py
    FGSM.py
checkpoint/
  .gitkeep
data/
  .gitkeep
model_architecture/
  cait.py
  CarliniNetwork.py
  MultiOutputSVM.py
  ResNet.py
  spiking_resnet_voter.py
  spiking_vgg_voter.py
  UNet.py
  VGG.py
.gitignore
AttackFactory.py
AttackRunner.py
AttackWrappersAdaptiveBlackBox.py
config.py
constants.py
DataLoaderGiant.py
main.py
ModelFactory.py
README.md
utils.py
</directory_structure>

<files>
This section contains the contents of the repository's files.

<file path="attacks/l0_attack/__init__.py">
# Import attack wrapper functions
from .L0_PGD import L0_PGD_AttackWrapper
from .L0_Sigma_PGD import L0_Sigma_PGD_AttackWrapper
from .L0_Linf_PGD import L0_Linf_PGD_AttackWrapper
</file>

<file path="attacks/l0_attack/L0_Linf_PGD.py">
import numpy as np
import torch
import torch.nn as nn
from . import utils
from . import L0_Utils

def L0_Linf_PGD_AttackWrapper(model, device, dataLoader, n_restarts, num_steps, step_size, sparsity, epsilon, random_start):
    
    model.eval()
    
    if random_start and n_restarts > 1:
            raise ValueError(
            f"Invalid parameter combination: random_start={random_start} and n_restarts={n_restarts}. "
            f"When using multiple restarts (n_restarts > 1), random_start should be False, "
            f"or use n_restarts=1 with random_start=True."
        )
    
    total_batches = len(dataLoader)
    total_samples = len(dataLoader.dataset)
    
    # First pass: Collect all original examples and labels
    x_tensor, y_tensor = utils.DataLoaderToTensor(dataLoader)
    all_original_examples, all_labels = utils.TensorToNumpy(x_tensor, y_tensor)
    
    # Initialize adversarial examples and robust accuracy
    all_adv_examples = np.copy(all_original_examples)
    all_latest_attempts = np.copy(all_original_examples)  # ------> Added to save failed adv examples as well
    pgd_adv_acc = None
    
    # Outer loop: Multiple restarts
    for counter in range(n_restarts):
        print(f"Restart {counter + 1}/{n_restarts}")
        
        if counter == 0:
            # Get initial predictions on clean images
            corr_pred = utils.get_predictions(model, all_original_examples, all_labels, device)
            pgd_adv_acc = np.copy(corr_pred)
        
        # Inner loop: Process each batch in this restart
        batch_start_idx = 0
        for batch_idx, (x_batch, y_batch) in enumerate(dataLoader):
            x_numpy, y_numpy = utils.TensorToNumpy(x_batch, y_batch)
            batch_size = x_numpy.shape[0]
            batch_end_idx = batch_start_idx + batch_size
            
            # Get the original samples for this batch
            x_nat = all_original_examples[batch_start_idx:batch_end_idx]
            y_nat = all_labels[batch_start_idx:batch_end_idx]
            
            # Perform attack on this batch
            x_batch_adv, curr_pgd_adv_acc = L0_Utils.perturb_L0_box(
                model, x_nat, y_nat, 
                np.maximum(-epsilon, -x_nat), 
                np.minimum(epsilon, 1.0 - x_nat), 
                sparsity, num_steps, step_size, device, random_start
            )
            
            # Update robust accuracy: take minimum (worst case) across restarts
            pgd_adv_acc[batch_start_idx:batch_end_idx] = np.minimum(
                pgd_adv_acc[batch_start_idx:batch_end_idx], 
                curr_pgd_adv_acc
            )
            
            # Update adversarial examples for samples that were successfully attacked
            mask = np.logical_not(curr_pgd_adv_acc)
            all_adv_examples[batch_start_idx:batch_end_idx][mask] = x_batch_adv[mask]

            # NEW: Always store latest attempt for ALL samples (will use for failed ones later)    ----> Added later to return failed adversarial
            all_latest_attempts[batch_start_idx:batch_end_idx] = x_batch_adv
            
            batch_start_idx = batch_end_idx

    # NEW: After all restarts, fill in failed samples with their latest attempts
    still_robust = pgd_adv_acc.astype(bool)  # True where attack never succeeded
    all_adv_examples[still_robust] = all_latest_attempts[still_robust]
    
    # Calculate overall statistics
    overall_robust_acc = np.sum(pgd_adv_acc) / total_samples * 100.0
    
    # Calculate pixels changed statistics
    pixels_changed = np.sum(np.amax(np.abs(all_adv_examples - all_original_examples) > 1e-10, axis=-1), axis=(1,2))
    
    # Calculate maximum perturbation across all samples
    max_perturbation = np.amax(np.abs(all_adv_examples - all_original_examples))

    # print('Pixels changed: ', pixels_changed)     # Uncomment it to print total pixel changes in each samples
    print(f"{'='*70}")
    print(f"Total samples processed: {total_samples}")
    print(f"Overall Robust Accuracy at {sparsity} pixels: {overall_robust_acc:.2f}%")
    print(f"Maximum perturbation size: {max_perturbation:.5f}")
    print(f"{'='*70}\n")

    # Print per-class robust accuracy using the new function
    # utils.print_per_class_robust_accuracy(all_labels, all_robust_acc)    # Uncomment to print classwise accuracy

    # Convert numpy arrays back to tensors using NumpyToTensor
    xAdv, yClean = utils.NumpyToTensor(all_adv_examples, all_labels)
    
    # Create and return adversarial dataLoader
    advLoader = utils.TensorToDataLoader(xAdv, yClean, transforms=None, batchSize=dataLoader.batch_size, randomizer=None)
     
    return advLoader
</file>

<file path="attacks/l0_attack/L0_PGD.py">
import numpy as np
import torch
import torch.nn as nn
from . import utils
from . import L0_Utils


def L0_PGD_AttackWrapper(model, device, dataLoader, n_restarts, num_steps, step_size, sparsity, random_start):

    model.eval()

    if random_start and n_restarts > 1:
            raise ValueError(
            f"Invalid parameter combination: random_start={random_start} and n_restarts={n_restarts}. "
            f"When using multiple restarts (n_restarts > 1), random_start should be False, "
            f"or use n_restarts=1 with random_start=True."
        )
    
    total_batches = len(dataLoader)
    total_samples = len(dataLoader.dataset)
    
    # First pass: Collect all original examples and labels
    x_tensor, y_tensor = utils.DataLoaderToTensor(dataLoader)
    all_original_examples, all_labels = utils.TensorToNumpy(x_tensor, y_tensor)
    
    # Initialize adversarial examples and robust accuracy
    all_adv_examples = np.copy(all_original_examples)
    all_latest_attempts = np.copy(all_original_examples)  # ------> Added to save failed adv examples as well
    pgd_adv_acc = None
    
    # Outer loop: Multiple restarts
    for counter in range(n_restarts):
        print(f"Restart {counter + 1}/{n_restarts}")
        
        if counter == 0:
            # Get initial predictions on clean images
            corr_pred = utils.get_predictions(model, all_original_examples, all_labels, device)
            pgd_adv_acc = np.copy(corr_pred)
        
        # Inner loop: Process each batch in this restart
        batch_start_idx = 0
        for batch_idx, (x_batch, y_batch) in enumerate(dataLoader):
            x_numpy, y_numpy = utils.TensorToNumpy(x_batch, y_batch)
            batch_size = x_numpy.shape[0]
            batch_end_idx = batch_start_idx + batch_size
            
            # Get the original samples for this batch
            x_nat = all_original_examples[batch_start_idx:batch_end_idx]
            y_nat = all_labels[batch_start_idx:batch_end_idx]
            
            # Perform attack on this batch
            x_batch_adv, curr_pgd_adv_acc = L0_Utils.perturb_L0_box(
                model, x_nat, y_nat, 
                -x_nat, 
                1.0 - x_nat, 
                sparsity, num_steps, step_size, device, random_start
            )
            
            # Update robust accuracy: take minimum (worst case) across restarts
            pgd_adv_acc[batch_start_idx:batch_end_idx] = np.minimum(
                pgd_adv_acc[batch_start_idx:batch_end_idx], 
                curr_pgd_adv_acc
            )
            
            # Update adversarial examples for samples that were successfully attacked
            mask = np.logical_not(curr_pgd_adv_acc)
            all_adv_examples[batch_start_idx:batch_end_idx][mask] = x_batch_adv[mask]

            # NEW: Always store latest attempt for ALL samples (will use for failed ones later)    ----> Added later to return failed adversarial
            all_latest_attempts[batch_start_idx:batch_end_idx] = x_batch_adv
            
            batch_start_idx = batch_end_idx

    # NEW: After all restarts, fill in failed samples with their latest attempts
    still_robust = pgd_adv_acc.astype(bool)  # True where attack never succeeded
    all_adv_examples[still_robust] = all_latest_attempts[still_robust]
    
    # Calculate overall statistics
    overall_robust_acc = np.sum(pgd_adv_acc) / total_samples * 100.0

    # Calculate pixels changed statistics
    pixels_changed = np.sum(np.amax(np.abs(all_adv_examples - all_original_examples) > 1e-10, axis=-1), axis=(1,2))

    # Calculate maximum perturbation across all samples
    max_perturbation = np.amax(np.abs(all_adv_examples - all_original_examples))

    # print('Pixels changed: ', pixels_changed)    # Uncomment it to print total pixel changes in each samples
    print(f"{'='*70}")
    print(f"Total samples processed: {total_samples}")
    print(f"Overall Robust Accuracy at {sparsity} pixels: {overall_robust_acc:.2f}%")
    print(f"Maximum perturbation size: {max_perturbation:.5f}")
    print(f"{'='*70}\n")

    # Print per-class robust accuracy using the new function
    # utils.print_per_class_robust_accuracy(all_labels, all_robust_acc)   # Uncomment to print classwise accuracy
    
    # Convert numpy arrays back to tensors using NumpyToTensor
    xAdv, yClean = utils.NumpyToTensor(all_adv_examples, all_labels)
    
    # Create and return adversarial dataLoader
    advLoader = utils.TensorToDataLoader(xAdv, yClean, transforms=None, batchSize=dataLoader.batch_size, randomizer=None)
     
    return advLoader
</file>

<file path="attacks/l0_attack/L0_Sigma_PGD.py">
import numpy as np
import torch
import torch.nn as nn
from . import utils
from . import L0_Utils

def L0_Sigma_PGD_AttackWrapper(model, device, dataLoader, n_restarts, num_steps, step_size, sparsity, kappa, random_start):
    
    model.eval()

    if random_start and n_restarts > 1:
            raise ValueError(
            f"Invalid parameter combination: random_start={random_start} and n_restarts={n_restarts}. "
            f"When using multiple restarts (n_restarts > 1), random_start should be False, "
            f"or use n_restarts=1 with random_start=True."
        )
    
    total_batches = len(dataLoader)
    total_samples = len(dataLoader.dataset)
    
    # First pass: Collect all original examples and labels
    x_tensor, y_tensor = utils.DataLoaderToTensor(dataLoader)
    all_original_examples, all_labels = utils.TensorToNumpy(x_tensor, y_tensor)
    
    # Calculate sigma map once for all samples (efficiency improvement)
    sigma = L0_Utils.sigma_map(all_original_examples)
    
    # Initialize adversarial examples and robust accuracy
    all_adv_examples = np.copy(all_original_examples)
    all_latest_attempts = np.copy(all_original_examples)  # ------> Added to save failed adv examples as well
    pgd_adv_acc = None
    
    # Outer loop: Multiple restarts
    for counter in range(n_restarts):
        print(f"Restart {counter + 1}/{n_restarts}")
        
        if counter == 0:
            # Get initial predictions on clean images
            corr_pred = utils.get_predictions(model, all_original_examples, all_labels, device)
            pgd_adv_acc = np.copy(corr_pred)
        
        # Inner loop: Process each batch in this restart
        batch_start_idx = 0
        for batch_idx, (x_batch, y_batch) in enumerate(dataLoader):
            x_numpy, y_numpy = utils.TensorToNumpy(x_batch, y_batch)
            batch_size = x_numpy.shape[0]
            batch_end_idx = batch_start_idx + batch_size
            
            # Get the original samples and sigma for this batch
            x_nat = all_original_examples[batch_start_idx:batch_end_idx]
            y_nat = all_labels[batch_start_idx:batch_end_idx]
            sigma_batch = sigma[batch_start_idx:batch_end_idx]
            
            # Perform attack on this batch
            x_batch_adv, curr_pgd_adv_acc = L0_Utils.perturb_L0_sigma(
                model, x_nat, y_nat, 
                sparsity, num_steps, step_size, 
                device, sigma_batch, kappa, random_start
            )
            
            # Update robust accuracy: take minimum (worst case) across restarts
            pgd_adv_acc[batch_start_idx:batch_end_idx] = np.minimum(
                pgd_adv_acc[batch_start_idx:batch_end_idx], 
                curr_pgd_adv_acc
            )
            
            # Update adversarial examples for samples that were successfully attacked
            mask = np.logical_not(curr_pgd_adv_acc)
            all_adv_examples[batch_start_idx:batch_end_idx][mask] = x_batch_adv[mask]

            # NEW: Always store latest attempt for ALL samples (will use for failed ones later)    ----> Added later to return failed adversarial
            all_latest_attempts[batch_start_idx:batch_end_idx] = x_batch_adv
            
            batch_start_idx = batch_end_idx

    # NEW: After all restarts, fill in failed samples with their latest attempts
    still_robust = pgd_adv_acc.astype(bool)  # True where attack never succeeded
    all_adv_examples[still_robust] = all_latest_attempts[still_robust]
    
    # Calculate overall statistics
    overall_robust_acc = np.sum(pgd_adv_acc) / total_samples * 100.0
    
    # Calculate pixels changed statistics
    pixels_changed = np.sum(np.amax(np.abs(all_adv_examples - all_original_examples) > 1e-10, axis=-1), axis=(1,2))
    
    # Calculate maximum perturbation across all samples
    max_perturbation = np.amax(np.abs(all_adv_examples - all_original_examples))
    
    # print('Pixels changed: ', pixels_changed)    # Uncomment it to print total pixel changes in each samples
    print(f"{'='*70}")
    print(f"Total samples processed: {total_samples}")
    print(f"Overall Robust Accuracy at {sparsity} pixels: {overall_robust_acc:.2f}%")
    print(f"Samples correctly classified after attack: {np.sum(pgd_adv_acc)}")
    print(f"Maximum perturbation size: {max_perturbation:.5f}")
    print(f"{'='*70}\n")

    # Print per-class robust accuracy using the new function
    # utils.print_per_class_robust_accuracy(all_labels, pgd_adv_acc)  # Uncomment to print classwise accuracy
    
    # Convert numpy arrays back to tensors using NumpyToTensor
    xAdv, yClean = utils.NumpyToTensor(all_adv_examples, all_labels)
    
    # Create and return adversarial dataLoader
    advLoader = utils.TensorToDataLoader(xAdv, yClean, transforms=None, batchSize=dataLoader.batch_size, randomizer=None)
     
    return advLoader
</file>

<file path="attacks/l0_attack/L0_Utils.py">
import numpy as np
import torch
from . import utils

def perturb_L0_box(model, x_nat, y_nat, lb, ub, sparsity, num_steps, step_size, device, random_start):
  if random_start == True:   
    x2 = x_nat + np.random.uniform(lb, ub, x_nat.shape)
    x2 = np.clip(x2, 0, 1)
  else:
    x2 = np.copy(x_nat)
      
  adv_not_found = np.ones(y_nat.shape)
  adv = np.zeros(x_nat.shape)

  for i in range(num_steps):
    if i > 0:
      pred, grad = utils.get_predictions_and_gradients(model, x2, y_nat, device)
      adv_not_found = np.minimum(adv_not_found, pred.astype(int))
      adv[np.logical_not(pred)] = np.copy(x2[np.logical_not(pred)])
      grad /= (1e-10 + np.sum(np.abs(grad), axis=(1,2,3), keepdims=True))
      x2 = np.add(x2, (np.random.random_sample(grad.shape)-0.5)*1e-12 + step_size * grad, casting='unsafe')
    x2 = x_nat + project_L0_box(x2 - x_nat, sparsity, lb, ub)

  # Fill in failed cases with final x2
  adv[adv_not_found.astype(bool)] = x2[adv_not_found.astype(bool)]    # ----> Added to return failed adversarial samples as well
    
  return adv, adv_not_found

def project_L0_box(y, k, lb, ub):
  x = np.copy(y)
  p1 = np.sum(x**2, axis=-1)
  p2 = np.minimum(np.minimum(ub - x, x - lb), 0)
  p2 = np.sum(p2**2, axis=-1)
  p3 = np.sort(np.reshape(p1-p2, [p2.shape[0],-1]))[:,-k]
  x = x*(np.logical_and(lb <=x, x <= ub)) + lb*(lb > x) + ub*(x > ub)
  x *= np.expand_dims((p1 - p2) >= p3.reshape([-1, 1, 1]), -1)
  return x

def sigma_map(x):
    ''' creates the sigma-map for the batch x '''
    sh = [4]
    sh.extend(x.shape)
    t = np.zeros(sh)
    t[0,:,:-1] = x[:,1:]
    t[0,:,-1] = x[:,-1]
    t[1,:,1:] = x[:,:-1]
    t[1,:,0] = x[:,0]
    t[2,:,:,:-1] = x[:,:,1:]
    t[2,:,:,-1] = x[:,:,-1]
    t[3,:,:,1:] = x[:,:,:-1]
    t[3,:,:,0] = x[:,:,0]

    mean1 = (t[0] + x + t[1]) / 3
    sd1 = np.sqrt(((t[0] - mean1) ** 2 + (x - mean1) ** 2 + (t[1] - mean1) ** 2) / 3)

    mean2 = (t[2] + x + t[3]) / 3
    sd2 = np.sqrt(((t[2] - mean2) ** 2 + (x - mean2) ** 2 + (t[3] - mean2) ** 2) / 3)

    sd = np.minimum(sd1, sd2)
    sd = np.sqrt(sd)

    return sd

def perturb_L0_sigma(model, x_nat, y_nat, sparsity, num_steps, step_size, device, sigma, kappa, random_start=True):
    if random_start == True:
        x2 = x_nat + np.random.uniform(-kappa, kappa, x_nat.shape)
        x2 = np.clip(x2, 0, 1)
    else:
        x2 = np.copy(x_nat)
    adv_not_found = np.ones(y_nat.shape)
    adv = np.zeros(x_nat.shape)

    for i in range(num_steps):
        if i > 0:
            pred, grad = utils.get_predictions_and_gradients(model, x2, y_nat, device)
            adv_not_found = np.minimum(adv_not_found, pred.astype(int))
            adv[np.logical_not(pred)] = np.copy(x2[np.logical_not(pred)])

            grad /= (1e-10 + np.sum(np.abs(grad), axis=(1,2,3), keepdims=True))
            x2 = np.add(x2, (np.random.random_sample(grad.shape) - 0.5) * 1e-12 + step_size * grad, casting='unsafe')

        x2 = project_L0_sigma(x2, sparsity, sigma, kappa, x_nat)

    # Fill in failed cases with final x2
    adv[adv_not_found.astype(bool)] = x2[adv_not_found.astype(bool)]    # Added to return failed adversarial samples as well

    return adv, adv_not_found

def project_L0_sigma(y, k, sigma, kappa, x_nat):
    x = np.copy(y)
    p1 = 1.0 / np.maximum(1e-12, sigma) * (x_nat > 0).astype(float) + 1e12 * (x_nat == 0).astype(float)
    p2 = 1.0 / np.maximum(1e-12, sigma) * (1.0 / np.maximum(1e-12, x_nat) - 1) * (x_nat > 0).astype(float) + \
         1e12 * (x_nat == 0).astype(float) + 1e12 * (sigma == 0).astype(float)
    lmbd_l = np.maximum(-kappa, np.amax(-p1, axis=-1, keepdims=True))
    lmbd_u = np.minimum(kappa, np.amin(p2, axis=-1, keepdims=True))
    
    lmbd_unconstr = np.sum((y - x_nat) * sigma * x_nat, axis=-1, keepdims=True) / \
                    np.maximum(1e-12, np.sum((sigma * x_nat) ** 2, axis=-1, keepdims=True))
    lmbd = np.maximum(lmbd_l, np.minimum(lmbd_unconstr, lmbd_u))
    
    p12 = np.sum((y - x_nat) ** 2, axis=-1, keepdims=True)
    p22 = np.sum((y - (1 + lmbd * sigma) * x_nat) ** 2, axis=-1, keepdims=True)
    p3 = np.sort(np.reshape(p12 - p22, [x.shape[0], -1]))[:, -k]
    
    x = x_nat + lmbd * sigma * x_nat * ((p12 - p22) >= p3.reshape([-1, 1, 1, 1]))
    
    return x
</file>

<file path="attacks/l0_attack/utils.py">
import torch
import torch.nn as nn
import numpy as np

#Convert a dataloader into x and y tensors 
def DataLoaderToTensor(dataLoader):
    #First check how many samples in the dataset
    numSamples = len(dataLoader.dataset) 
    sampleShape = GetOutputShape(dataLoader) #Get the output shape from the dataloader
    sampleIndex = 0
    #xData = torch.zeros(numSamples, sampleShape[0], sampleShape[1], sampleShape[2])
    xData = torch.zeros((numSamples,) + sampleShape) #Make it generic shape for non-image datasets
    yData = torch.zeros(numSamples)
    #Go through and process the data in batches 
    for i, (input, target) in enumerate(dataLoader):
        batchSize = input.shape[0] #Get the number of samples used in each batch
        #Save the samples from the batch in a separate tensor 
        for batchIndex in range(0, batchSize):
            xData[sampleIndex] = input[batchIndex]
            yData[sampleIndex] = target[batchIndex]
            sampleIndex = sampleIndex + 1 #increment the sample index 
    return xData, yData

#Convert a X and Y tensors into a dataloader
#Does not put any transforms with the data  
def TensorToDataLoader(xData, yData, transforms= None, batchSize=None, randomizer = None):
    if batchSize is None: #If no batch size put all the data through 
        batchSize = xData.shape[0]
    dataset = MyDataSet(xData, yData, transforms)
    if randomizer == None: #No randomizer
        dataLoader = torch.utils.data.DataLoader(dataset=dataset,  batch_size=batchSize, shuffle=False)
    else: #randomizer needed 
        train_sampler = torch.utils.data.RandomSampler(dataset)
        dataLoader = torch.utils.data.DataLoader(dataset=dataset,  batch_size=batchSize, sampler=train_sampler, shuffle=False)
    return dataLoader

def TensorToNumpy(x_tensor, y_tensor):
    x_numpy = x_tensor.cpu().numpy()
    x_numpy = x_numpy.transpose(0, 2, 3, 1)  # NCHW -> NHWC
    
    y_numpy = y_tensor.cpu().numpy()
    y_numpy = y_numpy.astype(np.int64)
    
    return x_numpy, y_numpy

def NumpyToTensor(x_numpy, y_numpy):
    # NHWC -> NCHW (reverse of TensorToNumpy)
    x_numpy = x_numpy.transpose(0, 3, 1, 2)
    
    x_tensor = torch.from_numpy(x_numpy).float()
    y_tensor = torch.from_numpy(y_numpy).long()
    
    return x_tensor, y_tensor

def get_predictions(model, x_nat, y_nat, device):
    x = torch.from_numpy(x_nat).permute(0, 3, 1, 2).float().to(device)
    y = torch.from_numpy(y_nat).to(device)
    with torch.no_grad():
        output = model(x)
    
    return (output.max(dim=-1)[1] == y).cpu().numpy()

def get_predictions_and_gradients(model, x_nat, y_nat, device):
    x = torch.from_numpy(x_nat).permute(0, 3, 1, 2).float().to(device)
    x.requires_grad_()
    y = torch.from_numpy(y_nat).to(device)

    with torch.enable_grad():
        output = model(x)

        # Cross Entropy loss function
        # loss = nn.CrossEntropyLoss()(output, y)

        # DLR Loss Function
        loss = dlr_loss(output, y).mean()  # Take mean to get scalar loss value

    grad = torch.autograd.grad(loss, x)[0]
    grad = grad.detach().permute(0, 2, 3, 1).cpu().numpy()

    pred = (output.detach().max(dim=-1)[1] == y).detach().cpu().numpy()

    return pred, grad

# Find the actual min and max pixel values in the dataset
def GetDataBounds(dataLoader, device):
    minVal = float('inf')
    maxVal = float('-inf')
    
    for xData, _ in dataLoader:
        xData = xData.to(device)
        batchMin = xData.min().item()
        batchMax = xData.max().item()
        
        if batchMin < minVal:
            minVal = batchMin
        if batchMax > maxVal:
            maxVal = batchMax
    
    return minVal, maxVal

#Validate using a dataloader 
def validateD(valLoader, model, device=None):
    #switch to evaluate mode
    model.eval()
    acc = 0
    batchTracker = 0
    with torch.no_grad():
        #Go through and process the data in batches 
        for i, (input, target) in enumerate(valLoader):
            sampleSize = input.shape[0] #Get the number of samples used in each batch
            batchTracker = batchTracker + sampleSize
            #print("Processing up to sample=", batchTracker)
            if device == None: #assume cuda
                inputVar = input.cuda()
            else:
                inputVar = input.to(device)
            #compute output
            output = model(inputVar)
            output = output.float()
            #Go through and check how many samples correctly identified
            for j in range(0, sampleSize):
                if output[j].argmax(axis=0) == target[j]:
                    acc = acc +1
    acc = acc / float(len(valLoader.dataset))
    return acc

def GetCorrectlyIdentifiedSamplesBalanced(model, totalSamplesRequired, dataLoader, numClasses, device=None):
    sampleShape = GetOutputShape(dataLoader)
    xData, yData = DataLoaderToTensor(dataLoader)
    #Basic error checking 
    if totalSamplesRequired % numClasses != 0:
        raise ValueError("The total number of samples in not evenly divisable by the number of classes.")
    #Get the number of samples needed for each class
    numSamplesPerClass = int(totalSamplesRequired/numClasses) 
    correctlyClassifiedSamples = torch.zeros((numClasses, numSamplesPerClass, sampleShape[0], sampleShape[1], sampleShape[2]))
    sanityCounter = torch.zeros((numClasses))
    #yPred = model.predict(xData)
    yPred = predictD(dataLoader, numClasses, model, device)
    a = 0
    for i in range(0, xData.shape[0]): #Go through every sample 
        a = a + 1
        predictedClass = yPred[i].argmax(axis=0)
        trueClass = yData[i]#.argmax(axis=0) 
        currentSavedCount = int(sanityCounter[int(trueClass)]) #Check how may samples we previously saved from this class
        #If the network predicts the sample correctly and we haven't saved enough samples from this class yet then save it
        if predictedClass == trueClass and currentSavedCount<numSamplesPerClass:
            correctlyClassifiedSamples[int(trueClass), currentSavedCount] = xData[i] #Save the sample 
            sanityCounter[int(trueClass)] = sanityCounter[int(trueClass)] + 1 #Add one to the count of saved samples for this class
    #Now we have gone through the entire network, make sure we have enough samples
    for c in range(0, numClasses):
        if sanityCounter[c] != numSamplesPerClass:
            raise ValueError("The network does not have enough correctly predicted samples for this class.")
    #Assume we have enough samples now, restore in a properly shaped array 
    xCorrect = torch.zeros((totalSamplesRequired, xData.shape[1], xData.shape[2], xData.shape[3]))
    yCorrect = torch.zeros((totalSamplesRequired))
    currentIndex = 0 #indexing for the final array
    for c in range(0, numClasses): #Go through each class
        for j in range(0, numSamplesPerClass): #For each sample in the class store it 
            xCorrect[currentIndex] = correctlyClassifiedSamples[c,j]
            yCorrect[currentIndex] = c
            #yCorrect[currentIndex, c] = 1.0
            currentIndex = currentIndex + 1 
    #return xCorrect, yCorrect
    cleanDataLoader = TensorToDataLoader(xCorrect, yCorrect, transforms = None, batchSize = dataLoader.batch_size, randomizer = None)
    return cleanDataLoader

#Get the output shape from the dataloader
def GetOutputShape(dataLoader):
    for i, (input, target) in enumerate(dataLoader):
        return input[0].shape

#Replicate TF's predict method behavior 
def predictD(dataLoader, numClasses, model, device=None):
    numSamples = len(dataLoader.dataset)
    yPred = torch.zeros(numSamples, numClasses)
    #switch to evaluate mode
    model.eval()
    indexer = 0
    batchTracker = 0
    with torch.no_grad():
        #Go through and process the data in batches 
        for i, (input, target) in enumerate(dataLoader):
            sampleSize = input.shape[0] #Get the number of samples used in each batch
            batchTracker = batchTracker + sampleSize
            #print("Processing up to sample=", batchTracker)
            if device == None:
                inputVar = input.cuda()
            else:
                inputVar = input.to(device)
            #compute output
            output = model(inputVar)
            output = output.float()
            for j in range(0, sampleSize):
                yPred[indexer] = output[j]
                indexer = indexer + 1 #update the indexer regardless 
    return yPred

def print_per_class_robust_accuracy(all_labels, all_robust_acc):
    unique_labels = np.unique(all_labels)
    
    print(f"\n{'='*70}")
    print(f"Per-Class Robust Accuracy:")
    print(f"{'='*70}\n")
    
    for label in unique_labels:
        # Get indices for this class
        class_indices = (all_labels == label)
        class_total = np.sum(class_indices)
        class_robust = np.sum(all_robust_acc[class_indices])
        class_robust_acc = (class_robust / class_total) * 100.0
        
        print(f"Class {int(label)}:")
        print(f"  Total samples: {class_total}")
        print(f"  Correctly classified after attack: {int(class_robust)}")
        print(f"  Robust Accuracy: {class_robust_acc:.2f}%")
        print(f"{'-'*70}")
    
    print(f"{'='*70}\n")

# DLR Loss Function
def dlr_loss(x, y):
    x_sorted, ind_sorted = x.sort(dim=1)
    ind = (ind_sorted[:, -1] == y).float()
    u = torch.arange(x.shape[0])
    return -(x[u, y] - x_sorted[:, -2] * ind - x_sorted[:, -1] * (1. - ind))

#Class to help with converting between dataloader and pytorch tensor 
class MyDataSet(torch.utils.data.Dataset):
    def __init__(self, x_tensor, y_tensor, transforms=None):
        self.x = x_tensor
        self.y = y_tensor
        self.transforms = transforms

    def __getitem__(self, index):
        if self.transforms is None: #No transform so return the data directly
            return (self.x[index], self.y[index])
        else: #Transform so apply it to the data before returning 
            return (self.transforms(self.x[index]), self.y[index])

    def __len__(self):
        return len(self.x)
</file>

<file path="attacks/l1_attack/__init__.py">
try:
    # prefer a class already named AutoAttack
    from .autoattack import AutoAttack
except ImportError:
    # fallback: export AutoAttackL1 under the public name AutoAttack
    from .autoattack import AutoAttackL1 as AutoAttack
</file>

<file path="attacks/l1_attack/autoattack.py">
# autoattack_l1.py
import math
import time
import numpy as np
import torch

from .other_utils import Logger
from . import checks
from .state import EvaluationState
from .autopgd_base import APGDAttack  # only dependency we need for attacks


class AutoAttackL1:
    """
    Minimal AutoAttack-style orchestrator that ONLY supports:
      - APGD on Cross-Entropy loss  ('apgd-ce')
      - APGD on DLR loss            ('apgd-dlr')
    Norm is fixed to 'L1'. No FAB / Square / targeted variants.

    Versions:
      - 'standard': runs apgd-ce then apgd-dlr, APGD n_restarts=5 (L1 default)
      - 'plus':     same attacks, but you can bump iterations/restarts if you want
      - 'rand':     same attacks with eot_iter=20 (randomized defenses)
      - 'custom':   respects attacks_to_run you pass in (subset of the two above)
    """

    def __init__(
        self,
        model,
        eps,
        seed=None,
        verbose=True,
        attacks_to_run=None,
        version="standard",
        device="cuda",
        log_path=None,
        n_iter=100,
        n_restarts=None,      # if None we'll pick good defaults below
        eot_iter=1,
        rho=0.75
    ):
        self.model = model
        self.norm = "L1"
        self.epsilon = eps
        self.seed = seed
        self.verbose = verbose
        self.version = version
        self.device = device
        self.logger = Logger(log_path)

        # attacks we allow
        self._ALL = ["apgd-ce", "apgd-dlr"]
        self.attacks_to_run = list(attacks_to_run) if attacks_to_run else []

        # APGD (single instance; we switch the loss between 'ce' and 'dlr')
        self.apgd = APGDAttack(
            self.model,
            n_restarts=5 if n_restarts is None else n_restarts,
            n_iter=n_iter,
            verbose=False,
            eps=self.epsilon,
            norm=self.norm,
            eot_iter=eot_iter,
            rho=rho,
            seed=self.seed,
            device=self.device,
            logger=self.logger,
        )

        # Map version presets (and validate attacks_to_run)
        self._configure_version()

    # ------------------------ helpers ------------------------

    def get_logits(self, x):
        return self.model(x)

    def get_seed(self):
        return time.time() if self.seed is None else self.seed

    def _configure_version(self):
        if self.version in ["standard", "plus", "rand"] and self.attacks_to_run:
            raise ValueError(
                "attacks_to_run will be overridden unless you use version='custom'"
            )

        if self.version == "standard":
            # Classic ordering
            self.attacks_to_run = ["apgd-ce", "apgd-dlr"]
            # APGD defaults for L1
            self.apgd.n_restarts = 5
            self.apgd.eot_iter = 1

        elif self.version == "plus":
            self.attacks_to_run = ["apgd-ce", "apgd-dlr"]
            self.apgd.n_restarts = 5  # bump to your taste
            self.apgd.eot_iter = 1

        elif self.version == "rand":
            self.attacks_to_run = ["apgd-ce", "apgd-dlr"]
            self.apgd.n_restarts = 1
            self.apgd.eot_iter = 20  # stochastic defenses

        elif self.version == "custom":
            # Keep only supported items, preserve user order
            if not self.attacks_to_run:
                raise ValueError("With version='custom', provide attacks_to_run.")
            unsupported = [a for a in self.attacks_to_run if a not in self._ALL]
            if unsupported:
                raise ValueError(f"Unsupported attacks in attacks_to_run: {unsupported}")
        else:
            raise ValueError(f"Unknown version: {self.version}")

        if self.verbose:
            self.logger.log(
                f"Configured version '{self.version}' with attacks: {', '.join(self.attacks_to_run)}"
            )

    # ------------------------ public API ------------------------

    @torch.no_grad()
    def clean_accuracy(self, x_orig, y_orig, bs=250):
        n_batches = math.ceil(x_orig.shape[0] / bs)
        acc = 0.0
        for counter in range(n_batches):
            x = x_orig[counter * bs : min((counter + 1) * bs, x_orig.shape[0])].clone().to(self.device)
            y = y_orig[counter * bs : min((counter + 1) * bs, x_orig.shape[0])].clone().to(self.device)
            output = self.get_logits(x)
            acc += (output.max(1)[1] == y).float().sum()

        if self.verbose:
            self.logger.log(f"clean accuracy: {acc / x_orig.shape[0]:.2%}")
        return acc.item() / x_orig.shape[0]

    def run_standard_evaluation(
        self,
        x_orig,
        y_orig,
        bs=250,
        return_labels=False,
        state_path=None,
    ):
        """
        Runs APGD-CE then APGD-DLR (or whatever is in attacks_to_run), only on
        currently robust points, like AutoAttack's choreography. Supports resuming via state.
        """
        # ----- state (optional resume) -----
        if state_path is not None and state_path.exists():
            state = EvaluationState.from_disk(state_path)
            if set(self.attacks_to_run) != state.attacks_to_run:
                raise ValueError(
                    "The state was created with a different set of attacks to run."
                )
            if self.verbose:
                self.logger.log(f"Restored state from {state_path}")
                self.logger.log(
                    "Since the state has been restored, only adversarials from the current run are returned."
                )
        else:
            state = EvaluationState(set(self.attacks_to_run), path=state_path)
            state.to_disk()
            if self.verbose and state_path is not None:
                self.logger.log(f"Created state in {state_path}")

        attacks_to_run = [a for a in self.attacks_to_run if a not in state.run_attacks]
        if self.verbose:
            self.logger.log(
                f"using {self.version} version including {', '.join(attacks_to_run)}."
            )
            if state.run_attacks:
                self.logger.log(f"{', '.join(state.run_attacks)} was/were already run.")

        # ----- pre-checks -----
        if self.version != "rand":
            checks.check_randomized(
                self.get_logits, x_orig[:bs].to(self.device), y_orig[:bs].to(self.device), bs=bs, logger=self.logger
            )
        n_cls = checks.check_range_output(self.get_logits, x_orig[:bs].to(self.device), logger=self.logger)
        checks.check_dynamic(self.model, x_orig[:bs].to(self.device), is_tf_model=False, logger=self.logger)
        # keep a minimal class-count sanity (no targeted logic here)
        if n_cls < 2:
            raise ValueError("Model appears to have < 2 classes from output range check.")

        # ----- initial clean evaluation -----
        with torch.no_grad():
            n_batches = int(np.ceil(x_orig.shape[0] / bs))
            if state.robust_flags is None:
                robust_flags = torch.zeros(x_orig.shape[0], dtype=torch.bool, device=x_orig.device)
                y_adv = torch.empty_like(y_orig)
                for batch_idx in range(n_batches):
                    start_idx = batch_idx * bs
                    end_idx = min((batch_idx + 1) * bs, x_orig.shape[0])
                    x = x_orig[start_idx:end_idx, :].clone().to(self.device)
                    y = y_orig[start_idx:end_idx].clone().to(self.device)
                    output = self.get_logits(x).max(dim=1)[1]
                    y_adv[start_idx:end_idx] = output
                    robust_flags[start_idx:end_idx] = y.eq(output)

                state.robust_flags = robust_flags
                robust_accuracy = torch.sum(robust_flags).item() / x_orig.shape[0]
                robust_accuracy_dict = {"clean": robust_accuracy}
                state.clean_accuracy = robust_accuracy
                if self.verbose:
                    self.logger.log(f"initial accuracy: {robust_accuracy:.2%}")
            else:
                robust_flags = state.robust_flags.to(x_orig.device)
                robust_accuracy = torch.sum(robust_flags).item() / x_orig.shape[0]
                robust_accuracy_dict = {"clean": state.clean_accuracy}
                if self.verbose:
                    self.logger.log(f"initial clean accuracy: {state.clean_accuracy:.2%}")
                    self.logger.log(
                        f"robust accuracy at the time of restoring the state: {robust_accuracy:.2%}"
                    )

            x_adv = x_orig.clone().detach()
            startt = time.time()

            # ----- main loop over attacks -----
            for attack in attacks_to_run:
                num_robust = int(torch.sum(robust_flags).item())
                if num_robust == 0:
                    break

                n_batches = int(np.ceil(num_robust / bs))
                robust_lin_idcs = torch.nonzero(robust_flags, as_tuple=False)
                if num_robust > 1:
                    robust_lin_idcs.squeeze_()

                for batch_idx in range(n_batches):
                    start_idx = batch_idx * bs
                    end_idx = min((batch_idx + 1) * bs, num_robust)
                    batch_datapoint_idcs = robust_lin_idcs[start_idx:end_idx]
                    if len(batch_datapoint_idcs.shape) > 1:
                        batch_datapoint_idcs.squeeze_(-1)

                    x = x_orig[batch_datapoint_idcs, :].clone().to(self.device)
                    y = y_orig[batch_datapoint_idcs].clone().to(self.device)
                    if len(x.shape) == 3:
                        x = x.unsqueeze(0)

                    # run APGD with desired loss
                    if attack == "apgd-ce":
                        self.apgd.loss = "ce"
                        self.apgd.seed = self.get_seed()
                        adv_curr = self.apgd.perturb(x, y)
                    elif attack == "apgd-dlr":
                        self.apgd.loss = "dlr"
                        self.apgd.seed = self.get_seed()
                        adv_curr = self.apgd.perturb(x, y)
                    else:
                        raise ValueError(f"Attack not supported: {attack}")

                    output = self.get_logits(adv_curr).max(dim=1)[1]
                    false_batch = ~y.eq(output).to(robust_flags.device)
                    non_robust_lin_idcs = batch_datapoint_idcs[false_batch]
                    robust_flags[non_robust_lin_idcs] = False
                    state.robust_flags = robust_flags

                    x_adv[non_robust_lin_idcs] = adv_curr[false_batch].detach().to(x_adv.device)
                    # store adversarial labels (optional; mirrors original behavior)
                    y_adv[non_robust_lin_idcs] = output[false_batch].detach().to(x_adv.device)

                    if self.verbose:
                        num_non_robust_batch = int(torch.sum(false_batch).item())
                        self.logger.log(
                            f"{attack} - {batch_idx + 1}/{n_batches} - {num_non_robust_batch} out of {x.shape[0]} successfully perturbed"
                        )

                robust_accuracy = torch.sum(robust_flags).item() / x_orig.shape[0]
                robust_accuracy_dict[attack] = robust_accuracy
                state.add_run_attack(attack)
                if self.verbose:
                    self.logger.log(
                        f"robust accuracy after {attack.upper()}: {robust_accuracy:.2%} (total time {time.time() - startt:.1f} s)"
                    )

            # ----- final checks and state save -----
            state.to_disk(force=True)
            if self.verbose:
                # L1 radius report
                res = (x_adv - x_orig).abs().reshape(x_orig.shape[0], -1).sum(dim=-1)
                self.logger.log(
                    f"max {self.norm} perturbation: {res.max():.5f}, "
                    f"nan in tensor: {(x_adv != x_adv).sum()}, max: {x_adv.max():.5f}, min: {x_adv.min():.5f}"
                )
                final_ra = torch.sum(robust_flags).item() / x_orig.shape[0]
                self.logger.log(f"robust accuracy: {final_ra:.2%}")

        return (x_adv, y_adv) if return_labels else x_adv

    # Convenience API mirroring upstream behavior
    def run_standard_evaluation_individual(self, x_orig, y_orig, bs=250, return_labels=False):
        if self.verbose:
            self.logger.log(f"using {self.version} version including {', '.join(self.attacks_to_run)}")

        adv = {}
        verbose_indiv = self.verbose
        self.verbose = False
        for attack in self.attacks_to_run:
            startt = time.time()
            x_adv, y_adv = self.run_standard_evaluation(x_orig, y_orig, bs=bs, return_labels=True)
            adv[attack] = (x_adv, y_adv) if return_labels else x_adv
            if verbose_indiv:
                acc_indiv = self.clean_accuracy(x_adv, y_orig, bs=bs)
                self.logger.log(
                    f"robust accuracy by {attack.upper()}\t {acc_indiv:.2%}\t (time attack: {time.time() - startt:.1f} s)"
                )
        return adv
</file>

<file path="attacks/l1_attack/autopgd_base.py">
## L1 separated version from original AutoAttack

import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from .other_utils import L0_norm, L1_norm
from .checks import check_zero_gradients


def L1_projection(x2, y2, eps1):
    '''
    x2: center of the L1 ball (bs x input_dim)
    y2: current perturbation (x2 + y2 is the point to be projected)
    eps1: radius of the L1 ball

    output: delta s.th. ||y2 + delta||_1 <= eps1
    and 0 <= x2 + y2 + delta <= 1
    '''
    x = x2.clone().float().view(x2.shape[0], -1) #original image x2 to 1D
    y = y2.clone().float().view(y2.shape[0], -1) #current perturbation y2 to 1D , perturbation vector d =u-x 
    sigma = y.clone().sign() # sign (=/-)of y vector s
    u = torch.min(1 - x - y, x + y) # compute y_i (gamma),   u=x+y? 
    u = torch.min(torch.zeros_like(y), u) #-u is exactly min(|u_i-x_i|, y_i)
    l = -torch.clone(y).abs() # -|y| corresponds to the point where the shrink lamda equals the perturbation magnitude |di|. #negative absolute current perturbation
    d = u.clone()
    #line 7
    bs, indbs = torch.sort(-torch.cat((u, l), 1), dim=1) # concatenates all breakpoints (u and l) into a single long vector and sorts them (bs). indbs are their indices
    bs2 = torch.cat((bs[:, 1:], torch.zeros(bs.shape[0], 1).to(bs.device)), 1) #bs2 is bs shifted left with a zero appended
 
    inu = 2*(indbs < u.shape[1]).float() - 1
    size1 = inu.cumsum(dim=1) #pre-calculates the slope of the piecewise linear function between each of the sorted breakpoints.

    #checking of easy case
    s1 = -u.sum(dim=1) # line 5

    c = eps1 - y.clone().abs().sum(dim=1) #remaining budget when starting from current perturbation y, c represents how much extra L1 we can still add to |y|_1 before hitting eps
    c5 = s1 + c < 0   #checks if this is over the budget (c5), S > eps (line 6), if exceeds budget need lamda_star
    c2 = c5.nonzero().squeeze(1) #check for hard case? 

    s = s1.unsqueeze(-1) + torch.cumsum((bs2 - bs) * size1, dim=1) #This vector pre-computes the value of the function f(lamda) at every breakpoint. S array is what will compare to eps to find where S crosses eps.

    if c2.nelement != 0:   # The if S > eps check is equivalent to it

      lb = torch.zeros_like(c2).float()
      ub = torch.ones_like(lb) *(bs.shape[1] - 1)

      nitermax = torch.ceil(torch.log2(torch.tensor(bs.shape[1]).float()))
      counter2 = torch.zeros_like(lb).long()
      counter = 0
       
      #implemented by concatenating breakpoints u and l, sorting them, building cumulative arrays, 
      #then binary-searching (lb/ub) to find the interval where the sum crosses eps, followed by computing alpha = lamda_star.
      while counter < nitermax:  # binary search to find the lamda_star  line 10-29 
        counter4 = torch.floor((lb + ub) / 2.)  # dividing the middle
        counter2 = counter4.type(torch.LongTensor)

        c8 = s[c2, counter2] + c[c2] < 0   #The c8 = s ... < 0 is the check that decides which half of the search space to discard.
        ind3 = c8.nonzero().squeeze(1)     #logic to update lb, ub 
        ind32 = (~c8).nonzero().squeeze(1)
        if ind3.nelement != 0:
            lb[ind3] = counter4[ind3]
        if ind32.nelement != 0:
            ub[ind32] = counter4[ind32]

        counter += 1

      lb2 = lb.long()  
      alpha = (-s[c2, lb2] -c[c2]) / size1[c2, lb2 + 1] + bs2[c2, lb2]  # Once the binary search finds the correct interval, it finds the exact value of lambda star using linear interpolation.
      d[c2] = -torch.min(torch.max(-u[c2], alpha.unsqueeze(-1)), -l[c2]) # This line uses the alpha to compute the final, corrected magnitudes for the perturbation for each pixel. 

    return (sigma * d).view(x2.shape)     #uses alpha and clamps to yield final d, which equals zi in magnitude.


class APGDAttack():
    """
    AutoPGD (L1-only, CE & DLR)
    https://arxiv.org/abs/2003.01690

    Kept function & method names/signatures identical to the upstream file,
    but restricted to L1 norm and untargeted CE / DLR losses.
    """

    def __init__(
            self,
            predict,
            n_iter=100,
            norm='L1',
            n_restarts=1,
            eps=None,
            seed=0,
            loss='ce',
            eot_iter=1,
            rho=.75,
            topk=None,
            verbose=False,
            device=None,
            use_largereps=False,
            is_tf_model=False,
            logger=None):
        """
        AutoPGD implementation in PyTorch (L1 + CE/DLR only).
        Names/args kept to preserve external API compatibility.
        """
        self.model = predict
        self.n_iter = n_iter
        self.eps = eps
        self.norm = norm
        self.n_restarts = n_restarts
        self.seed = seed
        self.loss = loss
        self.eot_iter = eot_iter
        self.thr_decr = rho
        self.topk = topk
        self.verbose = verbose
        self.device = device
        self.use_rs = True
        self.use_largereps = use_largereps
        self.n_iter_orig = n_iter + 0
        self.eps_orig = eps + 0. if eps is not None else None
        self.is_tf_model = is_tf_model  # kept
        self.y_target = None
        self.logger = logger

        # L1 only
        assert self.norm in ['L1'], 'This minimal build supports only L1.'
        assert self.eps is not None
        if self.is_tf_model:
            raise ValueError('TF models are not supported in this minimal L1 build.')

        # parameters for checkpoints
        self.n_iter_2 = max(int(0.22 * self.n_iter), 1)
        self.n_iter_min = max(int(0.06 * self.n_iter), 1)
        self.size_decr = max(int(0.03 * self.n_iter), 1)

    def init_hyperparam(self, x):
        if self.device is None:
            self.device = x.device
        self.orig_dim = list(x.shape[1:])
        self.ndims = len(self.orig_dim)
        if self.seed is None:
            self.seed = time.time()

    def check_shape(self, x):
        return x if len(x.shape) > 0 else x.unsqueeze(0)

    def normalize(self, x):
        # L1 normalize to avoid changing names
        try:
            t = x.abs().view(x.shape[0], -1).sum(dim=-1)
        except:
            t = x.abs().reshape([x.shape[0], -1]).sum(dim=-1)
        return x / (t.view(-1, *([1] * self.ndims)) + 1e-12)

   # def dlr_loss(self, x, y):
        #x_sorted, ind_sorted = x.sort(dim=1)
        #ind = (ind_sorted[:, -1] == y).float()
        #u = torch.arange(x.shape[0], device=x.device)
    def dlr_loss(self, x, y):
        """
        DLR loss with binary-safe 
        - C >= 3 (original AutoAttack):  -(z_y - z_k) / (z_(1) - z_(3) + eps)
        - C = 2 (binary):                -(z_y - z_k) 
        Returns: [B] tensor
        """
        x_sorted, ind_sorted = x.sort(dim=1)     # ascending: [:, -1]=max=z_(1), [:, -2]=z_(2), [:, -3]=z_(3)
        B, C = x.shape
        u = torch.arange(B, device=x.device)

        if C >= 3:
            # --- ORIGINAL (unchanged) ---
            ind = (ind_sorted[:, -1] == y).float()  # is y the argmax?
            return -(x[u, y] - x_sorted[:, -2] * ind - x_sorted[:, -1] * (1. - ind)) / (x_sorted[:, -1] - x_sorted[:, -3] + 1e-12)

        elif C == 2:
            # --- BINARY-SAFE --- aligned with paper
            # y must be 0/1; pick the other class logit directly
            other = y ^ 1  # same as (1 - y), but explicit for {0,1}
            zy = x[u, y]
            zk = x[u, other]
            return -(zy - zk)  
            #zmax = x_sorted[:, -1]                      # max logit
            #zmin = x_sorted[:,  0]                      # min logit
            #is_top = (ind_sorted[:, -1] == y)           # y is argmax?
            #zk = torch.where(is_top, zmin, zmax)        # second max if y is max; else max
            #zy = x[u, y]
            #return -(zy - zk) / (zmax - zmin + 1e-12)

        else:
            # C == 1 (degenerate): no meaningful margin
            return torch.zeros(B, device=x.device, dtype=x.dtype)




    def attack_single_run(self, x, y, x_init=None):
        if len(x.shape) < self.ndims:
            x = x.unsqueeze(0)
            y = y.unsqueeze(0)
            
        if x_init is not None:
            x_adv = x_init.clone()
            if self.verbose:
                print('[custom init] L1 perturbation {:.5f}'.format(
                    (x_adv - x).abs().view(x.shape[0], -1).sum(1).max()))
        
        elif getattr(self, "use_rs", True):
        # random L1 init (projection onto L1-ball  [0,1])
            t = torch.randn(x.shape, device=self.device).detach()
            delta = L1_projection(x, t, self.eps)
            x_adv = (x + t + delta)
        else:
        # deterministic start: exactly the clean image
            x_adv = x.clone()        

        x_adv = x_adv.clamp(0., 1.)
        x_best = x_adv.clone()
        x_best_adv = x_adv.clone()
        loss_steps = torch.zeros([self.n_iter, x.shape[0]], device=self.device)
        loss_best_steps = torch.zeros([self.n_iter + 1, x.shape[0]], device=self.device)
        acc_steps = torch.zeros_like(loss_best_steps)

        # CE / DLR only (PyTorch)
        if self.loss == 'ce':
            criterion_indiv = nn.CrossEntropyLoss(reduction='none')
        elif self.loss == 'dlr':
            criterion_indiv = self.dlr_loss
        else:
            raise ValueError('unknown loss (allowed: ce, dlr)')

        #  gradient at init  line 3–4: compute VL(x(i)) 
        x_adv.requires_grad_()
        grad = torch.zeros_like(x)
        for _ in range(self.eot_iter):
            with torch.enable_grad():
                logits = self.model(x_adv)
                loss_indiv = criterion_indiv(logits, y)
                loss = loss_indiv.sum()
            grad += torch.autograd.grad(loss, [x_adv])[0].detach()
        grad /= float(self.eot_iter)
        grad_best = grad.clone()

        if self.loss in ['dlr']:
            check_zero_gradients(grad, logger=self.logger)

        acc = logits.detach().max(1)[1] == y
        acc_steps[0] = acc + 0
        loss_best = loss_indiv.detach().clone()

        #alpha = 1.0  # L1 uses alpha=1 in upstream
        alpha = 0.1 # for voting dataset some cases
        step_size = alpha * self.eps * torch.ones([x.shape[0], *([1] * self.ndims)], device=self.device).detach()
        x_adv_old = x_adv.clone()
        k = max(int(.04 * self.n_iter), 1)
        n_fts = math.prod(self.orig_dim)

        # init sparsity schedule
        if x_init is None:
            topk = .2 * torch.ones([x.shape[0]], device=self.device)
            sp_old = n_fts * torch.ones_like(topk)
        else:
            topk = L0_norm(x_adv - x) / n_fts / 1.5
            sp_old = L0_norm(x_adv - x)

        adasp_redstep = 1.5
        adasp_minstep = 10.
        counter3 = 0
        loss_best_last_check = loss_best.clone()
        reduced_last_check = torch.ones_like(loss_best)

        u = torch.arange(x.shape[0], device=self.device)
        for i in range(self.n_iter): #Loop here line 4
            # gradient step (L1 sparse update + projection)
            with torch.no_grad():
                x_adv = x_adv.detach()
                grad2 = x_adv - x_adv_old
                x_adv_old = x_adv.clone()

                grad_topk = grad.abs().view(x.shape[0], -1).sort(-1)[0] #line 6 (compute sparse sign of gradient with k.d active coordinates) 
                topk_curr = torch.clamp((1. - topk) * n_fts, min=0, max=n_fts - 1).long()
                grad_topk = grad_topk[u, topk_curr].view(-1, *[1] * (len(x.shape) - 1))
                sparsegrad = grad * (grad.abs() >= grad_topk).float()
                x_adv_1 = x_adv + step_size * sparsegrad.sign() / (L1_norm(sparsegrad.sign(), keepdim=True) + 1e-10) #line 14 step to u, sparsegrad.sign() --S(VL,k.d) 

                delta_u = x_adv_1 - x  #line 15 -- proposed perturbation relative to clean x
                delta_p = L1_projection(x, delta_u, self.eps)
                #x_adv = (x + delta_u + delta_p)
                x_adv = (x_adv_1 + delta_p)
                #x_adv   = x + delta_p.clamp(0., 1.)
            # get gradient
            x_adv.requires_grad_()
            grad = torch.zeros_like(x)
            for _ in range(self.eot_iter):
                with torch.enable_grad():
                    logits = self.model(x_adv)
                    loss_indiv = criterion_indiv(logits, y)
                    loss = loss_indiv.sum()
                grad += torch.autograd.grad(loss, [x_adv])[0].detach()
            grad /= float(self.eot_iter)

            pred = logits.detach().max(1)[1] == y
            acc = torch.min(acc, pred)
            acc_steps[i + 1] = acc + 0
            ind_pred = (pred == 0).nonzero().squeeze()
            x_best_adv[ind_pred] = x_adv[ind_pred] + 0.
            if self.verbose:
                str_stats = ' - step size: {:.5f} - topk: {:.2f}'.format(step_size.mean(), topk.mean() * n_fts)
                print('[m] iteration: {} - best loss: {:.6f} - robust accuracy: {:.2%}{}'.format(
                    i, loss_best.sum(), acc.float().mean(), str_stats))

            # step-size & sparsity adaptation (L1)
            with torch.no_grad():
                y1 = loss_indiv.detach().clone()
                loss_steps[i] = y1 + 0
                ind = (y1 > loss_best).nonzero().squeeze() #track best loss/point 
                x_best[ind] = x_adv[ind].clone()
                grad_best[ind] = grad[ind].clone()
                loss_best[ind] = y1[ind] + 0
                loss_best_steps[i + 1] = loss_best + 0

                counter3 += 1
                if counter3 == k:  
                    sp_curr = L0_norm(x_best - x)  #  line 7 -- update k 
                    fl_redtopk = (sp_curr / sp_old) < .95 # did the sparsity change enough? 
                    topk = sp_curr / n_fts / 1.5   #equation 9 
                    step_size[fl_redtopk] = alpha * self.eps   # line 8 -- update n eta 
                    step_size[~fl_redtopk] /= adasp_redstep    # reduce as per equation 10 
                    step_size.clamp_(alpha * self.eps / adasp_minstep, alpha * self.eps)  #keep within eps and eps/10 
                    sp_old = sp_curr.clone()

                    x_adv[fl_redtopk] = x_best[fl_redtopk].clone()
                    grad[fl_redtopk] = grad_best[fl_redtopk].clone()

                    counter3 = 0

        return (x_best, acc, loss_best, x_best_adv)

    def perturb(self, x, y=None, best_loss=False, x_init=None):
        """
        :param x:           clean images
        :param y:           clean labels, if None we use the predicted labels
        :param best_loss:   if True return points with highest loss (not used by wrapper)
        :param x_init:      optional custom initialization
        """
        assert self.loss in ['ce', 'dlr']
        if y is not None and len(y.shape) == 0:
            x = x.unsqueeze(0)
            y = y.unsqueeze(0)
        self.init_hyperparam(x)

        x = x.detach().clone().float().to(self.device)
        y_pred = self.model(x).max(1)[1]
        if y is None:
            y = y_pred.detach().clone().long().to(self.device)  # not taken, y passed
        else:
            y = y.detach().clone().long().to(self.device)  #it is executed

        adv = x.clone()  # fill with adversarials
        acc = (y_pred == y)  # starts with correctly classified in this batch
        loss = -1e10 * torch.ones_like(acc).float()

        if self.verbose:
            print('-------------------------- ',
                'running {}-attack with epsilon {:.5f}'.format(self.norm, self.eps),
                '--------------------------')
            print('initial accuracy: {:.2%}'.format(acc.float().mean()))

        if self.use_largereps: # not executed, default is false
            epss = [3. * self.eps_orig, 2. * self.eps_orig, 1. * self.eps_orig]
            iters = [.3 * self.n_iter_orig, .3 * self.n_iter_orig, .4 * self.n_iter_orig]
            iters = [math.ceil(c) for c in iters]
            iters[-1] = self.n_iter_orig - sum(iters[:-1])

        if not best_loss: # True, as best loss pass yet (first run)
            torch.random.manual_seed(self.seed)
            torch.cuda.random.manual_seed(self.seed)

            for counter in range(self.n_restarts): # n_restart= 1 or 5 any value
                ind_to_fool = acc.nonzero().squeeze()
                if len(ind_to_fool.shape) == 0:
                    ind_to_fool = ind_to_fool.unsqueeze(0)
                if ind_to_fool.numel() != 0:
                    x_to_fool = x[ind_to_fool].clone()
                    y_to_fool = y[ind_to_fool].clone()

                    if not self.use_largereps: # True (as use_largereps is default False)
                        res_curr = self.attack_single_run(x_to_fool, y_to_fool, x_init=x_init) #x_init = NONE here, it calls attack_single_run() the main algorithm 1
                    else:
                        res_curr = self.decr_eps_pgd(x_to_fool, y_to_fool, epss, iters)  # not executed
                    best_curr, acc_curr, loss_curr, adv_curr = res_curr #do
                    ind_curr = (acc_curr == 0).nonzero().squeeze()  #do

                    acc[ind_to_fool[ind_curr]] = 0  # write back fooled examples to the correct original indices 
                    adv[ind_to_fool[ind_curr]] = adv_curr[ind_curr].clone()
                    if self.verbose:
                        print('restart {} - robust accuracy: {:.2%}'.format(
                            counter, acc.float().mean()))

            return adv # return this adv
        else: # not executed now, but executed when it is best loss
            adv_best = x.detach().clone()
            loss_best = torch.ones([x.shape[0]], device=self.device) * (-float('inf'))
            for counter in range(self.n_restarts):
                best_curr, _, loss_curr, _ = self.attack_single_run(x, y, x_init=x_init)
                ind_curr = (loss_curr > loss_best).nonzero().squeeze()
                adv_best[ind_curr] = best_curr[ind_curr] + 0.
                loss_best[ind_curr] = loss_curr[ind_curr] + 0.

                if self.verbose:
                    print('restart {} - loss: {:.5f}'.format(counter, loss_best.sum()))
            return adv_best

    def decr_eps_pgd(self, x, y, epss, iters, use_rs=True):
        assert len(epss) == len(iters)
        assert self.norm in ['L1']
        self.use_rs = False
        if not use_rs:
            x_init = None
        else:
            x_init = x + torch.randn_like(x)
            x_init += L1_projection(x, x_init - x, float(epss[0]))
        if self.verbose:
            print('total iter: {}'.format(sum(iters)))
        for eps, niter in zip(epss, iters):
            if self.verbose:
                print('using eps: {:.2f}'.format(eps))
            self.n_iter = niter + 0
            self.eps = eps + 0.
            if x_init is not None:
                x_init += L1_projection(x, x_init - x, float(eps))
            x_init, acc, loss, x_adv = self.attack_single_run(x, y, x_init=x_init)
        return (x_init, acc, loss, x_adv)


# Note: Linf/L2 branches have been removed intentionally
# to keep this file focused on L1 + CE/DLR while preserving the API functional.
</file>

<file path="attacks/l1_attack/checks.py">
import torch
import warnings
import math
import sys

from .other_utils import L2_norm

funcs = {'grad': 0,
    'backward': 0,
    '_make_grads': 0,
    }

checks_doc_path = 'flags_doc.md'


def check_randomized(model, x, y, bs=250, n=5, alpha=1e-4, logger=None):
    acc = []
    corrcl = []
    outputs = []
    with torch.no_grad():
        for _ in range(n):
            output = model(x)
            corrcl_curr = (output.max(1)[1] == y).sum().item()
            corrcl.append(corrcl_curr)
            outputs.append(output / (L2_norm(output, keepdim=True) + 1e-10))
    acc = [c != corrcl_curr for c in corrcl]
    max_diff = 0.
    for c in range(n - 1):
        for e in range(c + 1, n):
            diff = L2_norm(outputs[c] - outputs[e])
            max_diff = max(max_diff, diff.max().item())
    if any(acc) or max_diff > alpha:
        msg = 'it seems to be a randomized defense! Please use version="rand".' +             f' See {checks_doc_path} for details.'
        if logger is None:
            warnings.warn(Warning(msg))
        else:
            logger.log(f'Warning: {msg}')


def check_range_output(model, x, alpha=1e-5, logger=None):
    with torch.no_grad():
        output = model(x)
    fl = [output.max() < 1. + alpha, output.min() >  -alpha,
        ((output.sum(-1) - 1.).abs() < alpha).all()]
    if all(fl):
        msg = 'it seems that the output is a probability distribution,' +            ' please be sure that the logits are used!' +             f' See {checks_doc_path} for details.'
        if logger is None:
            warnings.warn(Warning(msg))
        else:
            logger.log(f'Warning: {msg}')
    return output.shape[-1]


def check_zero_gradients(grad, logger=None):
    z = grad.view(grad.shape[0], -1).abs().sum(-1)
    if (z == 0).any():
        msg = f'there are {(z == 0).sum()} points with zero gradient!' +             ' This might lead to unreliable evaluation with gradient-based attacks.' +             f' See {checks_doc_path} for details.'
        if logger is None:
            warnings.warn(Warning(msg))
        else:
            logger.log(f'Warning: {msg}')


def tracefunc(frame, event, args):
    if event == 'call' and frame.f_code.co_name in funcs.keys():
        funcs[frame.f_code.co_name] += 1


def check_dynamic(model, x, is_tf_model=False, logger=None):
    if is_tf_model:
        msg = 'the check for dynamic defenses is not currently supported'
    else:
        msg = None
        sys.settrace(tracefunc)
        model(x)
        sys.settrace(None)
        if any([c > 0 for c in funcs.values()]):
            msg = 'it seems to be a dynamic defense! The evaluation' +                 ' with AutoAttack might be insufficient.' +                 f' See {checks_doc_path} for details.'
    if msg is not None:
        if logger is None:
            warnings.warn(Warning(msg))
        else:
            logger.log(f'Warning: {msg}')


def check_n_classes(n_cls, attacks_to_run, apgd_targets=None, fab_targets=None, logger=None):
    """
    Minimal version:
      - Only sanity-checks DLR feasibility (n_cls > 2) when 'apgd-dlr' is requested.
      - Keeps the original signature for compatibility. Ignores targeted/FAB args.
    """
    msg = None
    if 'apgd-dlr' in attacks_to_run:
        if n_cls <= 2:
            msg = f'with only {n_cls} classes it is not possible to use the DLR loss!'
    if msg is not None:
        if logger is None:
            warnings.warn(Warning(msg))
        else:
            logger.log(f'Warning: {msg}')
</file>

<file path="attacks/l1_attack/other_utils.py">
import os
import collections.abc as container_abcs

import torch

class Logger():
    def __init__(self, log_path):
        self.log_path = log_path
        
    def log(self, str_to_log):
        print(str_to_log)
        if not self.log_path is None:
            with open(self.log_path, 'a') as f:
                f.write(str_to_log + '\n')
                f.flush()
            
def check_imgs(adv, x, norm):
    delta = (adv - x).view(adv.shape[0], -1)
    if norm == 'Linf':
        res = delta.abs().max(dim=1)[0]
    elif norm == 'L2':
        res = (delta ** 2).sum(dim=1).sqrt()
    elif norm == 'L1':
        res = delta.abs().sum(dim=1)

    str_det = 'max {} pert: {:.5f}, nan in imgs: {}, max in imgs: {:.5f}, min in imgs: {:.5f}'.format(
        norm, res.max(), (adv != adv).sum(), adv.max(), adv.min())
    print(str_det)
    
    return str_det

def L1_norm(x, keepdim=False):
    z = x.abs().view(x.shape[0], -1).sum(-1)
    if keepdim:
        z = z.view(-1, *[1]*(len(x.shape) - 1))
    return z

def L2_norm(x, keepdim=False):
    z = (x ** 2).view(x.shape[0], -1).sum(-1).sqrt()
    if keepdim:
        z = z.view(-1, *[1]*(len(x.shape) - 1))
    return z

def L0_norm(x):
    return (x != 0.).view(x.shape[0], -1).sum(-1)

def makedir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def zero_gradients(x):
    if isinstance(x, torch.Tensor):
        if x.grad is not None:
            x.grad.detach_()
            x.grad.zero_()
    elif isinstance(x, container_abcs.Iterable):
        for elem in x:
            zero_gradients(elem)
</file>

<file path="attacks/l1_attack/output.md">
This file is a merged representation of the entire codebase, combined into a single document by Repomix.

<file_summary>
This section contains a summary of this file.

<purpose>
This file contains a packed representation of the entire repository's contents.
It is designed to be easily consumable by AI systems for analysis, code review,
or other automated processes.
</purpose>

<file_format>
The content is organized as follows:
1. This summary section
2. Repository information
3. Directory structure
4. Repository files (if enabled)
5. Multiple file entries, each consisting of:
  - File path as an attribute
  - Full contents of the file
</file_format>

<usage_guidelines>
- This file should be treated as read-only. Any changes should be made to the
  original repository files, not this packed version.
- When processing this file, use the file path to distinguish
  between different files in the repository.
- Be aware that this file may contain sensitive information. Handle it with
  the same level of security as you would the original repository.
</usage_guidelines>

<notes>
- Some files may have been excluded based on .gitignore rules and Repomix's configuration
- Binary files are not included in this packed representation. Please refer to the Repository Structure section for a complete list of file paths, including binary files
- Files matching patterns in .gitignore are excluded
- Files matching default ignore patterns are excluded
- Files are sorted by Git change count (files with more changes are at the bottom)
</notes>

</file_summary>

<directory_structure>
__init__.py
autoattack.py
autopgd_base.py
checks.py
other_utils.py
state.py
</directory_structure>

<files>
This section contains the contents of the repository's files.

<file path="__init__.py">
try:
    # prefer a class already named AutoAttack
    from .autoattack import AutoAttack
except ImportError:
    # fallback: export AutoAttackL1 under the public name AutoAttack
    from .autoattack import AutoAttackL1 as AutoAttack
</file>

<file path="autoattack.py">
# autoattack_l1.py
import math
import time
import numpy as np
import torch

from .other_utils import Logger
from . import checks
from .state import EvaluationState
from .autopgd_base import APGDAttack  # only dependency we need for attacks


class AutoAttackL1:
    """
    Minimal AutoAttack-style orchestrator that ONLY supports:
      - APGD on Cross-Entropy loss  ('apgd-ce')
      - APGD on DLR loss            ('apgd-dlr')
    Norm is fixed to 'L1'. No FAB / Square / targeted variants.

    Versions:
      - 'standard': runs apgd-ce then apgd-dlr, APGD n_restarts=5 (L1 default)
      - 'plus':     same attacks, but you can bump iterations/restarts if you want
      - 'rand':     same attacks with eot_iter=20 (randomized defenses)
      - 'custom':   respects attacks_to_run you pass in (subset of the two above)
    """

    def __init__(
        self,
        model,
        eps,
        seed=None,
        verbose=True,
        attacks_to_run=None,
        version="standard",
        device="cuda",
        log_path=None,
        n_iter=100,
        n_restarts=None,      # if None we'll pick good defaults below
        eot_iter=1,
        rho=0.75
    ):
        self.model = model
        self.norm = "L1"
        self.epsilon = eps
        self.seed = seed
        self.verbose = verbose
        self.version = version
        self.device = device
        self.logger = Logger(log_path)

        # attacks we allow
        self._ALL = ["apgd-ce", "apgd-dlr"]
        self.attacks_to_run = list(attacks_to_run) if attacks_to_run else []

        # APGD (single instance; we switch the loss between 'ce' and 'dlr')
        self.apgd = APGDAttack(
            self.model,
            n_restarts=5 if n_restarts is None else n_restarts,
            n_iter=n_iter,
            verbose=False,
            eps=self.epsilon,
            norm=self.norm,
            eot_iter=eot_iter,
            rho=rho,
            seed=self.seed,
            device=self.device,
            logger=self.logger,
        )

        # Map version presets (and validate attacks_to_run)
        self._configure_version()

    # ------------------------ helpers ------------------------

    def get_logits(self, x):
        return self.model(x)

    def get_seed(self):
        return time.time() if self.seed is None else self.seed

    def _configure_version(self):
        if self.version in ["standard", "plus", "rand"] and self.attacks_to_run:
            raise ValueError(
                "attacks_to_run will be overridden unless you use version='custom'"
            )

        if self.version == "standard":
            # Classic ordering
            self.attacks_to_run = ["apgd-ce", "apgd-dlr"]
            # APGD defaults for L1
            self.apgd.n_restarts = 5
            self.apgd.eot_iter = 1

        elif self.version == "plus":
            self.attacks_to_run = ["apgd-ce", "apgd-dlr"]
            self.apgd.n_restarts = 5  # bump to your taste
            self.apgd.eot_iter = 1

        elif self.version == "rand":
            self.attacks_to_run = ["apgd-ce", "apgd-dlr"]
            self.apgd.n_restarts = 1
            self.apgd.eot_iter = 20  # stochastic defenses

        elif self.version == "custom":
            # Keep only supported items, preserve user order
            if not self.attacks_to_run:
                raise ValueError("With version='custom', provide attacks_to_run.")
            unsupported = [a for a in self.attacks_to_run if a not in self._ALL]
            if unsupported:
                raise ValueError(f"Unsupported attacks in attacks_to_run: {unsupported}")
        else:
            raise ValueError(f"Unknown version: {self.version}")

        if self.verbose:
            self.logger.log(
                f"Configured version '{self.version}' with attacks: {', '.join(self.attacks_to_run)}"
            )

    # ------------------------ public API ------------------------

    @torch.no_grad()
    def clean_accuracy(self, x_orig, y_orig, bs=250):
        n_batches = math.ceil(x_orig.shape[0] / bs)
        acc = 0.0
        for counter in range(n_batches):
            x = x_orig[counter * bs : min((counter + 1) * bs, x_orig.shape[0])].clone().to(self.device)
            y = y_orig[counter * bs : min((counter + 1) * bs, x_orig.shape[0])].clone().to(self.device)
            output = self.get_logits(x)
            acc += (output.max(1)[1] == y).float().sum()

        if self.verbose:
            self.logger.log(f"clean accuracy: {acc / x_orig.shape[0]:.2%}")
        return acc.item() / x_orig.shape[0]

    def run_standard_evaluation(
        self,
        x_orig,
        y_orig,
        bs=250,
        return_labels=False,
        state_path=None,
    ):
        """
        Runs APGD-CE then APGD-DLR (or whatever is in attacks_to_run), only on
        currently robust points, like AutoAttack's choreography. Supports resuming via state.
        """
        # ----- state (optional resume) -----
        if state_path is not None and state_path.exists():
            state = EvaluationState.from_disk(state_path)
            if set(self.attacks_to_run) != state.attacks_to_run:
                raise ValueError(
                    "The state was created with a different set of attacks to run."
                )
            if self.verbose:
                self.logger.log(f"Restored state from {state_path}")
                self.logger.log(
                    "Since the state has been restored, only adversarials from the current run are returned."
                )
        else:
            state = EvaluationState(set(self.attacks_to_run), path=state_path)
            state.to_disk()
            if self.verbose and state_path is not None:
                self.logger.log(f"Created state in {state_path}")

        attacks_to_run = [a for a in self.attacks_to_run if a not in state.run_attacks]
        if self.verbose:
            self.logger.log(
                f"using {self.version} version including {', '.join(attacks_to_run)}."
            )
            if state.run_attacks:
                self.logger.log(f"{', '.join(state.run_attacks)} was/were already run.")

        # ----- pre-checks -----
        if self.version != "rand":
            checks.check_randomized(
                self.get_logits, x_orig[:bs].to(self.device), y_orig[:bs].to(self.device), bs=bs, logger=self.logger
            )
        n_cls = checks.check_range_output(self.get_logits, x_orig[:bs].to(self.device), logger=self.logger)
        checks.check_dynamic(self.model, x_orig[:bs].to(self.device), is_tf_model=False, logger=self.logger)
        # keep a minimal class-count sanity (no targeted logic here)
        if n_cls < 2:
            raise ValueError("Model appears to have < 2 classes from output range check.")

        # ----- initial clean evaluation -----
        with torch.no_grad():
            n_batches = int(np.ceil(x_orig.shape[0] / bs))
            if state.robust_flags is None:
                robust_flags = torch.zeros(x_orig.shape[0], dtype=torch.bool, device=x_orig.device)
                y_adv = torch.empty_like(y_orig)
                for batch_idx in range(n_batches):
                    start_idx = batch_idx * bs
                    end_idx = min((batch_idx + 1) * bs, x_orig.shape[0])
                    x = x_orig[start_idx:end_idx, :].clone().to(self.device)
                    y = y_orig[start_idx:end_idx].clone().to(self.device)
                    output = self.get_logits(x).max(dim=1)[1]
                    y_adv[start_idx:end_idx] = output
                    robust_flags[start_idx:end_idx] = y.eq(output)

                state.robust_flags = robust_flags
                robust_accuracy = torch.sum(robust_flags).item() / x_orig.shape[0]
                robust_accuracy_dict = {"clean": robust_accuracy}
                state.clean_accuracy = robust_accuracy
                if self.verbose:
                    self.logger.log(f"initial accuracy: {robust_accuracy:.2%}")
            else:
                robust_flags = state.robust_flags.to(x_orig.device)
                robust_accuracy = torch.sum(robust_flags).item() / x_orig.shape[0]
                robust_accuracy_dict = {"clean": state.clean_accuracy}
                if self.verbose:
                    self.logger.log(f"initial clean accuracy: {state.clean_accuracy:.2%}")
                    self.logger.log(
                        f"robust accuracy at the time of restoring the state: {robust_accuracy:.2%}"
                    )

            x_adv = x_orig.clone().detach()
            startt = time.time()

            # ----- main loop over attacks -----
            for attack in attacks_to_run:
                num_robust = int(torch.sum(robust_flags).item())
                if num_robust == 0:
                    break

                n_batches = int(np.ceil(num_robust / bs))
                robust_lin_idcs = torch.nonzero(robust_flags, as_tuple=False)
                if num_robust > 1:
                    robust_lin_idcs.squeeze_()

                for batch_idx in range(n_batches):
                    start_idx = batch_idx * bs
                    end_idx = min((batch_idx + 1) * bs, num_robust)
                    batch_datapoint_idcs = robust_lin_idcs[start_idx:end_idx]
                    if len(batch_datapoint_idcs.shape) > 1:
                        batch_datapoint_idcs.squeeze_(-1)

                    x = x_orig[batch_datapoint_idcs, :].clone().to(self.device)
                    y = y_orig[batch_datapoint_idcs].clone().to(self.device)
                    if len(x.shape) == 3:
                        x = x.unsqueeze(0)

                    # run APGD with desired loss
                    if attack == "apgd-ce":
                        self.apgd.loss = "ce"
                        self.apgd.seed = self.get_seed()
                        adv_curr = self.apgd.perturb(x, y)
                    elif attack == "apgd-dlr":
                        self.apgd.loss = "dlr"
                        self.apgd.seed = self.get_seed()
                        adv_curr = self.apgd.perturb(x, y)
                    else:
                        raise ValueError(f"Attack not supported: {attack}")

                    output = self.get_logits(adv_curr).max(dim=1)[1]
                    false_batch = ~y.eq(output).to(robust_flags.device)
                    non_robust_lin_idcs = batch_datapoint_idcs[false_batch]
                    robust_flags[non_robust_lin_idcs] = False
                    state.robust_flags = robust_flags

                    x_adv[non_robust_lin_idcs] = adv_curr[false_batch].detach().to(x_adv.device)
                    # store adversarial labels (optional; mirrors original behavior)
                    y_adv[non_robust_lin_idcs] = output[false_batch].detach().to(x_adv.device)

                    if self.verbose:
                        num_non_robust_batch = int(torch.sum(false_batch).item())
                        self.logger.log(
                            f"{attack} - {batch_idx + 1}/{n_batches} - {num_non_robust_batch} out of {x.shape[0]} successfully perturbed"
                        )

                robust_accuracy = torch.sum(robust_flags).item() / x_orig.shape[0]
                robust_accuracy_dict[attack] = robust_accuracy
                state.add_run_attack(attack)
                if self.verbose:
                    self.logger.log(
                        f"robust accuracy after {attack.upper()}: {robust_accuracy:.2%} (total time {time.time() - startt:.1f} s)"
                    )

            # ----- final checks and state save -----
            state.to_disk(force=True)
            if self.verbose:
                # L1 radius report
                res = (x_adv - x_orig).abs().reshape(x_orig.shape[0], -1).sum(dim=-1)
                self.logger.log(
                    f"max {self.norm} perturbation: {res.max():.5f}, "
                    f"nan in tensor: {(x_adv != x_adv).sum()}, max: {x_adv.max():.5f}, min: {x_adv.min():.5f}"
                )
                final_ra = torch.sum(robust_flags).item() / x_orig.shape[0]
                self.logger.log(f"robust accuracy: {final_ra:.2%}")

        return (x_adv, y_adv) if return_labels else x_adv

    # Convenience API mirroring upstream behavior
    def run_standard_evaluation_individual(self, x_orig, y_orig, bs=250, return_labels=False):
        if self.verbose:
            self.logger.log(f"using {self.version} version including {', '.join(self.attacks_to_run)}")

        adv = {}
        verbose_indiv = self.verbose
        self.verbose = False
        for attack in self.attacks_to_run:
            startt = time.time()
            x_adv, y_adv = self.run_standard_evaluation(x_orig, y_orig, bs=bs, return_labels=True)
            adv[attack] = (x_adv, y_adv) if return_labels else x_adv
            if verbose_indiv:
                acc_indiv = self.clean_accuracy(x_adv, y_orig, bs=bs)
                self.logger.log(
                    f"robust accuracy by {attack.upper()}\t {acc_indiv:.2%}\t (time attack: {time.time() - startt:.1f} s)"
                )
        return adv
</file>

<file path="autopgd_base.py">
## L1 separated version from original AutoAttack

import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from .other_utils import L0_norm, L1_norm
from .checks import check_zero_gradients


def L1_projection(x2, y2, eps1):
    '''
    x2: center of the L1 ball (bs x input_dim)
    y2: current perturbation (x2 + y2 is the point to be projected)
    eps1: radius of the L1 ball

    output: delta s.th. ||y2 + delta||_1 <= eps1
    and 0 <= x2 + y2 + delta <= 1
    '''
    x = x2.clone().float().view(x2.shape[0], -1) #original image x2 to 1D
    y = y2.clone().float().view(y2.shape[0], -1) #current perturbation y2 to 1D , perturbation vector d =u-x 
    sigma = y.clone().sign() # sign (=/-)of y vector s
    u = torch.min(1 - x - y, x + y) # compute y_i (gamma),   u=x+y? 
    u = torch.min(torch.zeros_like(y), u) #-u is exactly min(|u_i-x_i|, y_i)
    l = -torch.clone(y).abs() # -|y| corresponds to the point where the shrink lamda equals the perturbation magnitude |di|. #negative absolute current perturbation
    d = u.clone()
    #line 7
    bs, indbs = torch.sort(-torch.cat((u, l), 1), dim=1) # concatenates all breakpoints (u and l) into a single long vector and sorts them (bs). indbs are their indices
    bs2 = torch.cat((bs[:, 1:], torch.zeros(bs.shape[0], 1).to(bs.device)), 1) #bs2 is bs shifted left with a zero appended
 
    inu = 2*(indbs < u.shape[1]).float() - 1
    size1 = inu.cumsum(dim=1) #pre-calculates the slope of the piecewise linear function between each of the sorted breakpoints.

    #checking of easy case
    s1 = -u.sum(dim=1) # line 5

    c = eps1 - y.clone().abs().sum(dim=1) #remaining budget when starting from current perturbation y, c represents how much extra L1 we can still add to |y|_1 before hitting eps
    c5 = s1 + c < 0   #checks if this is over the budget (c5), S > eps (line 6), if exceeds budget need lamda_star
    c2 = c5.nonzero().squeeze(1) #check for hard case? 

    s = s1.unsqueeze(-1) + torch.cumsum((bs2 - bs) * size1, dim=1) #This vector pre-computes the value of the function f(lamda) at every breakpoint. S array is what will compare to eps to find where S crosses eps.

    if c2.nelement != 0:   # The if S > eps check is equivalent to it

      lb = torch.zeros_like(c2).float()
      ub = torch.ones_like(lb) *(bs.shape[1] - 1)

      nitermax = torch.ceil(torch.log2(torch.tensor(bs.shape[1]).float()))
      counter2 = torch.zeros_like(lb).long()
      counter = 0
       
      #implemented by concatenating breakpoints u and l, sorting them, building cumulative arrays, 
      #then binary-searching (lb/ub) to find the interval where the sum crosses eps, followed by computing alpha = lamda_star.
      while counter < nitermax:  # binary search to find the lamda_star  line 10-29 
        counter4 = torch.floor((lb + ub) / 2.)  # dividing the middle
        counter2 = counter4.type(torch.LongTensor)

        c8 = s[c2, counter2] + c[c2] < 0   #The c8 = s ... < 0 is the check that decides which half of the search space to discard.
        ind3 = c8.nonzero().squeeze(1)     #logic to update lb, ub 
        ind32 = (~c8).nonzero().squeeze(1)
        if ind3.nelement != 0:
            lb[ind3] = counter4[ind3]
        if ind32.nelement != 0:
            ub[ind32] = counter4[ind32]

        counter += 1

      lb2 = lb.long()  
      alpha = (-s[c2, lb2] -c[c2]) / size1[c2, lb2 + 1] + bs2[c2, lb2]  # Once the binary search finds the correct interval, it finds the exact value of lambda star using linear interpolation.
      d[c2] = -torch.min(torch.max(-u[c2], alpha.unsqueeze(-1)), -l[c2]) # This line uses the alpha to compute the final, corrected magnitudes for the perturbation for each pixel. 

    return (sigma * d).view(x2.shape)     #uses alpha and clamps to yield final d, which equals zi in magnitude.


class APGDAttack():
    """
    AutoPGD (L1-only, CE & DLR)
    https://arxiv.org/abs/2003.01690

    Kept function & method names/signatures identical to the upstream file,
    but restricted to L1 norm and untargeted CE / DLR losses.
    """

    def __init__(
            self,
            predict,
            n_iter=100,
            norm='L1',
            n_restarts=1,
            eps=None,
            seed=0,
            loss='ce',
            eot_iter=1,
            rho=.75,
            topk=None,
            verbose=False,
            device=None,
            use_largereps=False,
            is_tf_model=False,
            logger=None):
        """
        AutoPGD implementation in PyTorch (L1 + CE/DLR only).
        Names/args kept to preserve external API compatibility.
        """
        self.model = predict
        self.n_iter = n_iter
        self.eps = eps
        self.norm = norm
        self.n_restarts = n_restarts
        self.seed = seed
        self.loss = loss
        self.eot_iter = eot_iter
        self.thr_decr = rho
        self.topk = topk
        self.verbose = verbose
        self.device = device
        self.use_rs = True
        self.use_largereps = use_largereps
        self.n_iter_orig = n_iter + 0
        self.eps_orig = eps + 0. if eps is not None else None
        self.is_tf_model = is_tf_model  # kept
        self.y_target = None
        self.logger = logger

        # L1 only
        assert self.norm in ['L1'], 'This minimal build supports only L1.'
        assert self.eps is not None
        if self.is_tf_model:
            raise ValueError('TF models are not supported in this minimal L1 build.')

        # parameters for checkpoints
        self.n_iter_2 = max(int(0.22 * self.n_iter), 1)
        self.n_iter_min = max(int(0.06 * self.n_iter), 1)
        self.size_decr = max(int(0.03 * self.n_iter), 1)

    def init_hyperparam(self, x):
        if self.device is None:
            self.device = x.device
        self.orig_dim = list(x.shape[1:])
        self.ndims = len(self.orig_dim)
        if self.seed is None:
            self.seed = time.time()

    def check_shape(self, x):
        return x if len(x.shape) > 0 else x.unsqueeze(0)

    def normalize(self, x):
        # L1 normalize to avoid changing names
        try:
            t = x.abs().view(x.shape[0], -1).sum(dim=-1)
        except:
            t = x.abs().reshape([x.shape[0], -1]).sum(dim=-1)
        return x / (t.view(-1, *([1] * self.ndims)) + 1e-12)

   # def dlr_loss(self, x, y):
        #x_sorted, ind_sorted = x.sort(dim=1)
        #ind = (ind_sorted[:, -1] == y).float()
        #u = torch.arange(x.shape[0], device=x.device)
    def dlr_loss(self, x, y):
        """
        DLR loss with binary-safe 
        - C >= 3 (original AutoAttack):  -(z_y - z_k) / (z_(1) - z_(3) + eps)
        - C = 2 (binary):                -(z_y - z_k) 
        Returns: [B] tensor
        """
        x_sorted, ind_sorted = x.sort(dim=1)     # ascending: [:, -1]=max=z_(1), [:, -2]=z_(2), [:, -3]=z_(3)
        B, C = x.shape
        u = torch.arange(B, device=x.device)

        if C >= 3:
            # --- ORIGINAL (unchanged) ---
            ind = (ind_sorted[:, -1] == y).float()  # is y the argmax?
            return -(x[u, y] - x_sorted[:, -2] * ind - x_sorted[:, -1] * (1. - ind)) / (x_sorted[:, -1] - x_sorted[:, -3] + 1e-12)

        elif C == 2:
            # --- BINARY-SAFE --- aligned with paper
            # y must be 0/1; pick the other class logit directly
            other = y ^ 1  # same as (1 - y), but explicit for {0,1}
            zy = x[u, y]
            zk = x[u, other]
            return -(zy - zk)  
            #zmax = x_sorted[:, -1]                      # max logit
            #zmin = x_sorted[:,  0]                      # min logit
            #is_top = (ind_sorted[:, -1] == y)           # y is argmax?
            #zk = torch.where(is_top, zmin, zmax)        # second max if y is max; else max
            #zy = x[u, y]
            #return -(zy - zk) / (zmax - zmin + 1e-12)

        else:
            # C == 1 (degenerate): no meaningful margin
            return torch.zeros(B, device=x.device, dtype=x.dtype)




    def attack_single_run(self, x, y, x_init=None):
        if len(x.shape) < self.ndims:
            x = x.unsqueeze(0)
            y = y.unsqueeze(0)
            
        if x_init is not None:
            x_adv = x_init.clone()
            if self.verbose:
                print('[custom init] L1 perturbation {:.5f}'.format(
                    (x_adv - x).abs().view(x.shape[0], -1).sum(1).max()))
        
        elif getattr(self, "use_rs", True):
        # random L1 init (projection onto L1-ball  [0,1])
            t = torch.randn(x.shape, device=self.device).detach()
            delta = L1_projection(x, t, self.eps)
            x_adv = (x + t + delta)
        else:
        # deterministic start: exactly the clean image
            x_adv = x.clone()        

        x_adv = x_adv.clamp(0., 1.)
        x_best = x_adv.clone()
        x_best_adv = x_adv.clone()
        loss_steps = torch.zeros([self.n_iter, x.shape[0]], device=self.device)
        loss_best_steps = torch.zeros([self.n_iter + 1, x.shape[0]], device=self.device)
        acc_steps = torch.zeros_like(loss_best_steps)

        # CE / DLR only (PyTorch)
        if self.loss == 'ce':
            criterion_indiv = nn.CrossEntropyLoss(reduction='none')
        elif self.loss == 'dlr':
            criterion_indiv = self.dlr_loss
        else:
            raise ValueError('unknown loss (allowed: ce, dlr)')

        #  gradient at init  line 3–4: compute VL(x(i)) 
        x_adv.requires_grad_()
        grad = torch.zeros_like(x)
        for _ in range(self.eot_iter):
            with torch.enable_grad():
                logits = self.model(x_adv)
                loss_indiv = criterion_indiv(logits, y)
                loss = loss_indiv.sum()
            grad += torch.autograd.grad(loss, [x_adv])[0].detach()
        grad /= float(self.eot_iter)
        grad_best = grad.clone()

        if self.loss in ['dlr']:
            check_zero_gradients(grad, logger=self.logger)

        acc = logits.detach().max(1)[1] == y
        acc_steps[0] = acc + 0
        loss_best = loss_indiv.detach().clone()

        #alpha = 1.0  # L1 uses alpha=1 in upstream
        alpha = 0.1 # for voting dataset some cases
        step_size = alpha * self.eps * torch.ones([x.shape[0], *([1] * self.ndims)], device=self.device).detach()
        x_adv_old = x_adv.clone()
        k = max(int(.04 * self.n_iter), 1)
        n_fts = math.prod(self.orig_dim)

        # init sparsity schedule
        if x_init is None:
            topk = .2 * torch.ones([x.shape[0]], device=self.device)
            sp_old = n_fts * torch.ones_like(topk)
        else:
            topk = L0_norm(x_adv - x) / n_fts / 1.5
            sp_old = L0_norm(x_adv - x)

        adasp_redstep = 1.5
        adasp_minstep = 10.
        counter3 = 0
        loss_best_last_check = loss_best.clone()
        reduced_last_check = torch.ones_like(loss_best)

        u = torch.arange(x.shape[0], device=self.device)
        for i in range(self.n_iter): #Loop here line 4
            # gradient step (L1 sparse update + projection)
            with torch.no_grad():
                x_adv = x_adv.detach()
                grad2 = x_adv - x_adv_old
                x_adv_old = x_adv.clone()

                grad_topk = grad.abs().view(x.shape[0], -1).sort(-1)[0] #line 6 (compute sparse sign of gradient with k.d active coordinates) 
                topk_curr = torch.clamp((1. - topk) * n_fts, min=0, max=n_fts - 1).long()
                grad_topk = grad_topk[u, topk_curr].view(-1, *[1] * (len(x.shape) - 1))
                sparsegrad = grad * (grad.abs() >= grad_topk).float()
                x_adv_1 = x_adv + step_size * sparsegrad.sign() / (L1_norm(sparsegrad.sign(), keepdim=True) + 1e-10) #line 14 step to u, sparsegrad.sign() --S(VL,k.d) 

                delta_u = x_adv_1 - x  #line 15 -- proposed perturbation relative to clean x
                delta_p = L1_projection(x, delta_u, self.eps)
                #x_adv = (x + delta_u + delta_p)
                x_adv = (x_adv_1 + delta_p)
                #x_adv   = x + delta_p.clamp(0., 1.)
            # get gradient
            x_adv.requires_grad_()
            grad = torch.zeros_like(x)
            for _ in range(self.eot_iter):
                with torch.enable_grad():
                    logits = self.model(x_adv)
                    loss_indiv = criterion_indiv(logits, y)
                    loss = loss_indiv.sum()
                grad += torch.autograd.grad(loss, [x_adv])[0].detach()
            grad /= float(self.eot_iter)

            pred = logits.detach().max(1)[1] == y
            acc = torch.min(acc, pred)
            acc_steps[i + 1] = acc + 0
            ind_pred = (pred == 0).nonzero().squeeze()
            x_best_adv[ind_pred] = x_adv[ind_pred] + 0.
            if self.verbose:
                str_stats = ' - step size: {:.5f} - topk: {:.2f}'.format(step_size.mean(), topk.mean() * n_fts)
                print('[m] iteration: {} - best loss: {:.6f} - robust accuracy: {:.2%}{}'.format(
                    i, loss_best.sum(), acc.float().mean(), str_stats))

            # step-size & sparsity adaptation (L1)
            with torch.no_grad():
                y1 = loss_indiv.detach().clone()
                loss_steps[i] = y1 + 0
                ind = (y1 > loss_best).nonzero().squeeze() #track best loss/point 
                x_best[ind] = x_adv[ind].clone()
                grad_best[ind] = grad[ind].clone()
                loss_best[ind] = y1[ind] + 0
                loss_best_steps[i + 1] = loss_best + 0

                counter3 += 1
                if counter3 == k:  
                    sp_curr = L0_norm(x_best - x)  #  line 7 -- update k 
                    fl_redtopk = (sp_curr / sp_old) < .95 # did the sparsity change enough? 
                    topk = sp_curr / n_fts / 1.5   #equation 9 
                    step_size[fl_redtopk] = alpha * self.eps   # line 8 -- update n eta 
                    step_size[~fl_redtopk] /= adasp_redstep    # reduce as per equation 10 
                    step_size.clamp_(alpha * self.eps / adasp_minstep, alpha * self.eps)  #keep within eps and eps/10 
                    sp_old = sp_curr.clone()

                    x_adv[fl_redtopk] = x_best[fl_redtopk].clone()
                    grad[fl_redtopk] = grad_best[fl_redtopk].clone()

                    counter3 = 0

        return (x_best, acc, loss_best, x_best_adv)

    def perturb(self, x, y=None, best_loss=False, x_init=None):
        """
        :param x:           clean images
        :param y:           clean labels, if None we use the predicted labels
        :param best_loss:   if True return points with highest loss (not used by wrapper)
        :param x_init:      optional custom initialization
        """
        assert self.loss in ['ce', 'dlr']
        if y is not None and len(y.shape) == 0:
            x = x.unsqueeze(0)
            y = y.unsqueeze(0)
        self.init_hyperparam(x)

        x = x.detach().clone().float().to(self.device)
        y_pred = self.model(x).max(1)[1]
        if y is None:
            y = y_pred.detach().clone().long().to(self.device)  # not taken, y passed
        else:
            y = y.detach().clone().long().to(self.device)  #it is executed

        adv = x.clone()  # fill with adversarials
        acc = (y_pred == y)  # starts with correctly classified in this batch
        loss = -1e10 * torch.ones_like(acc).float()

        if self.verbose:
            print('-------------------------- ',
                'running {}-attack with epsilon {:.5f}'.format(self.norm, self.eps),
                '--------------------------')
            print('initial accuracy: {:.2%}'.format(acc.float().mean()))

        if self.use_largereps: # not executed, default is false
            epss = [3. * self.eps_orig, 2. * self.eps_orig, 1. * self.eps_orig]
            iters = [.3 * self.n_iter_orig, .3 * self.n_iter_orig, .4 * self.n_iter_orig]
            iters = [math.ceil(c) for c in iters]
            iters[-1] = self.n_iter_orig - sum(iters[:-1])

        if not best_loss: # True, as best loss pass yet (first run)
            torch.random.manual_seed(self.seed)
            torch.cuda.random.manual_seed(self.seed)

            for counter in range(self.n_restarts): # n_restart= 1 or 5 any value
                ind_to_fool = acc.nonzero().squeeze()
                if len(ind_to_fool.shape) == 0:
                    ind_to_fool = ind_to_fool.unsqueeze(0)
                if ind_to_fool.numel() != 0:
                    x_to_fool = x[ind_to_fool].clone()
                    y_to_fool = y[ind_to_fool].clone()

                    if not self.use_largereps: # True (as use_largereps is default False)
                        res_curr = self.attack_single_run(x_to_fool, y_to_fool, x_init=x_init) #x_init = NONE here, it calls attack_single_run() the main algorithm 1
                    else:
                        res_curr = self.decr_eps_pgd(x_to_fool, y_to_fool, epss, iters)  # not executed
                    best_curr, acc_curr, loss_curr, adv_curr = res_curr #do
                    ind_curr = (acc_curr == 0).nonzero().squeeze()  #do

                    acc[ind_to_fool[ind_curr]] = 0  # write back fooled examples to the correct original indices 
                    adv[ind_to_fool[ind_curr]] = adv_curr[ind_curr].clone()
                    if self.verbose:
                        print('restart {} - robust accuracy: {:.2%}'.format(
                            counter, acc.float().mean()))

            return adv # return this adv
        else: # not executed now, but executed when it is best loss
            adv_best = x.detach().clone()
            loss_best = torch.ones([x.shape[0]], device=self.device) * (-float('inf'))
            for counter in range(self.n_restarts):
                best_curr, _, loss_curr, _ = self.attack_single_run(x, y, x_init=x_init)
                ind_curr = (loss_curr > loss_best).nonzero().squeeze()
                adv_best[ind_curr] = best_curr[ind_curr] + 0.
                loss_best[ind_curr] = loss_curr[ind_curr] + 0.

                if self.verbose:
                    print('restart {} - loss: {:.5f}'.format(counter, loss_best.sum()))
            return adv_best

    def decr_eps_pgd(self, x, y, epss, iters, use_rs=True):
        assert len(epss) == len(iters)
        assert self.norm in ['L1']
        self.use_rs = False
        if not use_rs:
            x_init = None
        else:
            x_init = x + torch.randn_like(x)
            x_init += L1_projection(x, x_init - x, float(epss[0]))
        if self.verbose:
            print('total iter: {}'.format(sum(iters)))
        for eps, niter in zip(epss, iters):
            if self.verbose:
                print('using eps: {:.2f}'.format(eps))
            self.n_iter = niter + 0
            self.eps = eps + 0.
            if x_init is not None:
                x_init += L1_projection(x, x_init - x, float(eps))
            x_init, acc, loss, x_adv = self.attack_single_run(x, y, x_init=x_init)
        return (x_init, acc, loss, x_adv)


# Note: Linf/L2 branches have been removed intentionally
# to keep this file focused on L1 + CE/DLR while preserving the API functional.
</file>

<file path="checks.py">
import torch
import warnings
import math
import sys

from .other_utils import L2_norm

funcs = {'grad': 0,
    'backward': 0,
    '_make_grads': 0,
    }

checks_doc_path = 'flags_doc.md'


def check_randomized(model, x, y, bs=250, n=5, alpha=1e-4, logger=None):
    acc = []
    corrcl = []
    outputs = []
    with torch.no_grad():
        for _ in range(n):
            output = model(x)
            corrcl_curr = (output.max(1)[1] == y).sum().item()
            corrcl.append(corrcl_curr)
            outputs.append(output / (L2_norm(output, keepdim=True) + 1e-10))
    acc = [c != corrcl_curr for c in corrcl]
    max_diff = 0.
    for c in range(n - 1):
        for e in range(c + 1, n):
            diff = L2_norm(outputs[c] - outputs[e])
            max_diff = max(max_diff, diff.max().item())
    if any(acc) or max_diff > alpha:
        msg = 'it seems to be a randomized defense! Please use version="rand".' +             f' See {checks_doc_path} for details.'
        if logger is None:
            warnings.warn(Warning(msg))
        else:
            logger.log(f'Warning: {msg}')


def check_range_output(model, x, alpha=1e-5, logger=None):
    with torch.no_grad():
        output = model(x)
    fl = [output.max() < 1. + alpha, output.min() >  -alpha,
        ((output.sum(-1) - 1.).abs() < alpha).all()]
    if all(fl):
        msg = 'it seems that the output is a probability distribution,' +            ' please be sure that the logits are used!' +             f' See {checks_doc_path} for details.'
        if logger is None:
            warnings.warn(Warning(msg))
        else:
            logger.log(f'Warning: {msg}')
    return output.shape[-1]


def check_zero_gradients(grad, logger=None):
    z = grad.view(grad.shape[0], -1).abs().sum(-1)
    if (z == 0).any():
        msg = f'there are {(z == 0).sum()} points with zero gradient!' +             ' This might lead to unreliable evaluation with gradient-based attacks.' +             f' See {checks_doc_path} for details.'
        if logger is None:
            warnings.warn(Warning(msg))
        else:
            logger.log(f'Warning: {msg}')


def tracefunc(frame, event, args):
    if event == 'call' and frame.f_code.co_name in funcs.keys():
        funcs[frame.f_code.co_name] += 1


def check_dynamic(model, x, is_tf_model=False, logger=None):
    if is_tf_model:
        msg = 'the check for dynamic defenses is not currently supported'
    else:
        msg = None
        sys.settrace(tracefunc)
        model(x)
        sys.settrace(None)
        if any([c > 0 for c in funcs.values()]):
            msg = 'it seems to be a dynamic defense! The evaluation' +                 ' with AutoAttack might be insufficient.' +                 f' See {checks_doc_path} for details.'
    if msg is not None:
        if logger is None:
            warnings.warn(Warning(msg))
        else:
            logger.log(f'Warning: {msg}')


def check_n_classes(n_cls, attacks_to_run, apgd_targets=None, fab_targets=None, logger=None):
    """
    Minimal version:
      - Only sanity-checks DLR feasibility (n_cls > 2) when 'apgd-dlr' is requested.
      - Keeps the original signature for compatibility. Ignores targeted/FAB args.
    """
    msg = None
    if 'apgd-dlr' in attacks_to_run:
        if n_cls <= 2:
            msg = f'with only {n_cls} classes it is not possible to use the DLR loss!'
    if msg is not None:
        if logger is None:
            warnings.warn(Warning(msg))
        else:
            logger.log(f'Warning: {msg}')
</file>

<file path="other_utils.py">
import os
import collections.abc as container_abcs

import torch

class Logger():
    def __init__(self, log_path):
        self.log_path = log_path
        
    def log(self, str_to_log):
        print(str_to_log)
        if not self.log_path is None:
            with open(self.log_path, 'a') as f:
                f.write(str_to_log + '\n')
                f.flush()
            
def check_imgs(adv, x, norm):
    delta = (adv - x).view(adv.shape[0], -1)
    if norm == 'Linf':
        res = delta.abs().max(dim=1)[0]
    elif norm == 'L2':
        res = (delta ** 2).sum(dim=1).sqrt()
    elif norm == 'L1':
        res = delta.abs().sum(dim=1)

    str_det = 'max {} pert: {:.5f}, nan in imgs: {}, max in imgs: {:.5f}, min in imgs: {:.5f}'.format(
        norm, res.max(), (adv != adv).sum(), adv.max(), adv.min())
    print(str_det)
    
    return str_det

def L1_norm(x, keepdim=False):
    z = x.abs().view(x.shape[0], -1).sum(-1)
    if keepdim:
        z = z.view(-1, *[1]*(len(x.shape) - 1))
    return z

def L2_norm(x, keepdim=False):
    z = (x ** 2).view(x.shape[0], -1).sum(-1).sqrt()
    if keepdim:
        z = z.view(-1, *[1]*(len(x.shape) - 1))
    return z

def L0_norm(x):
    return (x != 0.).view(x.shape[0], -1).sum(-1)

def makedir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def zero_gradients(x):
    if isinstance(x, torch.Tensor):
        if x.grad is not None:
            x.grad.detach_()
            x.grad.zero_()
    elif isinstance(x, container_abcs.Iterable):
        for elem in x:
            zero_gradients(elem)
</file>

<file path="state.py">
import json
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Optional, Set
import warnings

import torch


@dataclass
class EvaluationState:
    _attacks_to_run: Set[str]
    path: Optional[Path] = None
    _run_attacks: Set[str] = field(default_factory=set)
    _robust_flags: Optional[torch.Tensor] = None
    _last_saved: datetime = datetime(1, 1, 1)
    _SAVE_TIMEOUT: int = 60
    _clean_accuracy: float = float("nan")

    def to_disk(self, force: bool = False) -> None:
        seconds_since_last_save = (datetime.now() -
                                   self._last_saved).total_seconds()
        if self.path is None or (seconds_since_last_save < self._SAVE_TIMEOUT
                                 and not force):
            return
        self._last_saved = datetime.now()
        d = asdict(self)
        if self.robust_flags is not None:
            d["_robust_flags"] = d["_robust_flags"].cpu().tolist()
        d["_run_attacks"] = list(self._run_attacks)
        with self.path.open("w", ) as f:
            json.dump(d, f, default=str)

    @classmethod
    def from_disk(cls, path: Path) -> "EvaluationState":
        with path.open("r") as f:
            d = json.load(f)
        d["_robust_flags"] = torch.tensor(d["_robust_flags"], dtype=torch.bool)
        d["path"] = Path(d["path"])
        if path != d["path"]:
            warnings.warn(
                UserWarning(
                    "The given path is different from the one found in the state file."
                ))
        d["_last_saved"] = datetime.fromisoformat(d["_last_saved"])
        return cls(**d)

    @property
    def robust_flags(self) -> Optional[torch.Tensor]:
        return self._robust_flags

    @robust_flags.setter
    def robust_flags(self, robust_flags: torch.Tensor) -> None:
        self._robust_flags = robust_flags
        self.to_disk(force=True)

    @property
    def run_attacks(self) -> Set[str]:
        return self._run_attacks

    def add_run_attack(self, attack: str) -> None:
        self._run_attacks.add(attack)
        self.to_disk()
        
    @property
    def attacks_to_run(self) -> Set[str]:
        return self._attacks_to_run
    
    @attacks_to_run.setter
    def attacks_to_run(self, _: Set[str]) -> None:
        raise ValueError("attacks_to_run cannot be set outside of the constructor")

    @property
    def clean_accuracy(self) -> float:
        return self._clean_accuracy

    @clean_accuracy.setter
    def clean_accuracy(self, accuracy) -> None:
        self._clean_accuracy = accuracy
        self.to_disk(force=True)

    @property
    def robust_accuracy(self) -> float:
        if self.robust_flags is None:
            raise ValueError("robust_flags is not set yet. Start the attack first.")
        if self.attacks_to_run - self.run_attacks:
            warnings.warn("You are checking `robust_accuracy` before all the attacks"
                          " have been run.")
        return self.robust_flags.float().mean().item()
</file>

</files>
</file>

<file path="attacks/l1_attack/state.py">
import json
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Optional, Set
import warnings

import torch


@dataclass
class EvaluationState:
    _attacks_to_run: Set[str]
    path: Optional[Path] = None
    _run_attacks: Set[str] = field(default_factory=set)
    _robust_flags: Optional[torch.Tensor] = None
    _last_saved: datetime = datetime(1, 1, 1)
    _SAVE_TIMEOUT: int = 60
    _clean_accuracy: float = float("nan")

    def to_disk(self, force: bool = False) -> None:
        seconds_since_last_save = (datetime.now() -
                                   self._last_saved).total_seconds()
        if self.path is None or (seconds_since_last_save < self._SAVE_TIMEOUT
                                 and not force):
            return
        self._last_saved = datetime.now()
        d = asdict(self)
        if self.robust_flags is not None:
            d["_robust_flags"] = d["_robust_flags"].cpu().tolist()
        d["_run_attacks"] = list(self._run_attacks)
        with self.path.open("w", ) as f:
            json.dump(d, f, default=str)

    @classmethod
    def from_disk(cls, path: Path) -> "EvaluationState":
        with path.open("r") as f:
            d = json.load(f)
        d["_robust_flags"] = torch.tensor(d["_robust_flags"], dtype=torch.bool)
        d["path"] = Path(d["path"])
        if path != d["path"]:
            warnings.warn(
                UserWarning(
                    "The given path is different from the one found in the state file."
                ))
        d["_last_saved"] = datetime.fromisoformat(d["_last_saved"])
        return cls(**d)

    @property
    def robust_flags(self) -> Optional[torch.Tensor]:
        return self._robust_flags

    @robust_flags.setter
    def robust_flags(self, robust_flags: torch.Tensor) -> None:
        self._robust_flags = robust_flags
        self.to_disk(force=True)

    @property
    def run_attacks(self) -> Set[str]:
        return self._run_attacks

    def add_run_attack(self, attack: str) -> None:
        self._run_attacks.add(attack)
        self.to_disk()
        
    @property
    def attacks_to_run(self) -> Set[str]:
        return self._attacks_to_run
    
    @attacks_to_run.setter
    def attacks_to_run(self, _: Set[str]) -> None:
        raise ValueError("attacks_to_run cannot be set outside of the constructor")

    @property
    def clean_accuracy(self) -> float:
        return self._clean_accuracy

    @clean_accuracy.setter
    def clean_accuracy(self, accuracy) -> None:
        self._clean_accuracy = accuracy
        self.to_disk(force=True)

    @property
    def robust_accuracy(self) -> float:
        if self.robust_flags is None:
            raise ValueError("robust_flags is not set yet. Start the attack first.")
        if self.attacks_to_run - self.run_attacks:
            warnings.warn("You are checking `robust_accuracy` before all the attacks"
                          " have been run.")
        return self.robust_flags.float().mean().item()
</file>

<file path="attacks/l2_attack/autopgd_base.py">
# Copyright (c) 2020-present, Francesco Croce
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree
#

import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import random

from .other_utils import L0_norm, L1_norm, L2_norm
from .checks import check_zero_gradients

from . import DataManagerPytorch as DMP


def L1_projection(x2, y2, eps1):
    '''
    x2: center of the L1 ball (bs x input_dim)
    y2: current perturbation (x2 + y2 is the point to be projected)
    eps1: radius of the L1 ball

    output: delta s.th. ||y2 + delta||_1 <= eps1
    and 0 <= x2 + y2 + delta <= 1
    '''

    x = x2.clone().float().view(x2.shape[0], -1)
    y = y2.clone().float().view(y2.shape[0], -1)
    sigma = y.clone().sign()
    u = torch.min(1 - x - y, x + y)
    #u = torch.min(u, epsinf - torch.clone(y).abs())
    u = torch.min(torch.zeros_like(y), u)
    l = -torch.clone(y).abs()
    d = u.clone()
    
    bs, indbs = torch.sort(-torch.cat((u, l), 1), dim=1)
    bs2 = torch.cat((bs[:, 1:], torch.zeros(bs.shape[0], 1).to(bs.device)), 1)
    
    inu = 2*(indbs < u.shape[1]).float() - 1
    size1 = inu.cumsum(dim=1)
    
    s1 = -u.sum(dim=1)
    
    c = eps1 - y.clone().abs().sum(dim=1)
    c5 = s1 + c < 0
    c2 = c5.nonzero().squeeze(1)
    
    s = s1.unsqueeze(-1) + torch.cumsum((bs2 - bs) * size1, dim=1)
    
    if c2.nelement != 0:
    
      lb = torch.zeros_like(c2).float()
      ub = torch.ones_like(lb) *(bs.shape[1] - 1)
      
      #print(c2.shape, lb.shape)
      
      nitermax = torch.ceil(torch.log2(torch.tensor(bs.shape[1]).float()))
      counter2 = torch.zeros_like(lb).long()
      counter = 0
          
      while counter < nitermax:
        counter4 = torch.floor((lb + ub) / 2.)
        counter2 = counter4.type(torch.LongTensor)
        
        c8 = s[c2, counter2] + c[c2] < 0
        ind3 = c8.nonzero().squeeze(1)
        ind32 = (~c8).nonzero().squeeze(1)
        #print(ind3.shape)
        if ind3.nelement != 0:
            lb[ind3] = counter4[ind3]
        if ind32.nelement != 0:
            ub[ind32] = counter4[ind32]
        
        #print(lb, ub)
        counter += 1
        
      lb2 = lb.long()
      alpha = (-s[c2, lb2] -c[c2]) / size1[c2, lb2 + 1] + bs2[c2, lb2]
      d[c2] = -torch.min(torch.max(-u[c2], alpha.unsqueeze(-1)), -l[c2])
    
    return (sigma * d).view(x2.shape)





class APGDAttack():
    """
    AutoPGD
    https://arxiv.org/abs/2003.01690

    :param predict:       forward pass function
    :param norm:          Lp-norm of the attack ('Linf', 'L2', 'L0' supported)
    :param n_restarts:    number of random restarts
    :param n_iter:        number of iterations
    :param eps:           bound on the norm of perturbations
    :param seed:          random seed for the starting point
    :param loss:          loss to optimize ('ce', 'dlr' supported)
    :param eot_iter:      iterations for Expectation over Trasformation
    :param rho:           parameter for decreasing the step size
    """

    def __init__(
            self,
            predict,
            n_iter=100,
            norm='Linf',
            n_restarts=1,
            eps=None,
            seed=0,
            loss='ce',
            eot_iter=1,
            rho=.75,
            topk=None,
            verbose=False,
            device=None,
            use_largereps=False,
            is_tf_model=False,
            logger=None):
        """
        AutoPGD implementation in PyTorch
        """
        
        self.model = predict
        self.n_iter = n_iter
        self.eps = eps
        self.norm = norm
        self.n_restarts = n_restarts
        self.seed = seed
        self.loss = loss
        self.eot_iter = eot_iter
        self.thr_decr = rho
        self.topk = topk
        self.verbose = verbose
        self.device = device
        self.use_rs = True
        #self.init_point = None
        self.use_largereps = use_largereps
        #self.larger_epss = None
        #self.iters = None
        self.n_iter_orig = n_iter + 0
        self.eps_orig = eps + 0.
        self.is_tf_model = is_tf_model
        self.y_target = None
        self.logger = logger

        assert self.norm in ['Linf', 'L2', 'L1']
        assert not self.eps is None

        ### set parameters for checkpoints
        self.n_iter_2 = max(int(0.22 * self.n_iter), 1)
        self.n_iter_min = max(int(0.06 * self.n_iter), 1)
        self.size_decr = max(int(0.03 * self.n_iter), 1)

    def init_hyperparam(self, x):

        if self.device is None:
            self.device = x.device
        self.orig_dim = list(x.shape[1:])
        self.ndims = len(self.orig_dim)
        if self.seed is None:
            self.seed = time.time()

    def check_oscillation(self, x, j, k, y5, k3=0.75):
        t = torch.zeros(x.shape[1]).to(self.device)
        for counter5 in range(k):
          t += (x[j - counter5] > x[j - counter5 - 1]).float()

        return (t <= k * k3 * torch.ones_like(t)).float()

    def check_shape(self, x):
        return x if len(x.shape) > 0 else x.unsqueeze(0)

    def normalize(self, x):
        if self.norm == 'Linf':
            t = x.abs().view(x.shape[0], -1).max(1)[0]

        elif self.norm == 'L2':
            t = (x ** 2).view(x.shape[0], -1).sum(-1).sqrt()

        elif self.norm == 'L1':
            try:
                t = x.abs().view(x.shape[0], -1).sum(dim=-1)
            except:
                t = x.abs().reshape([x.shape[0], -1]).sum(dim=-1)

        return x / (t.view(-1, *([1] * self.ndims)) + 1e-12)

    #def dlr_loss(self, x, y):
    #    x_sorted, ind_sorted = x.sort(dim=1)
    #    ind = (ind_sorted[:, -1] == y).float()
    #    u = torch.arange(x.shape[0])

     #   return -(x[u, y] - x_sorted[:, -2] * ind - x_sorted[:, -1] * (
      #      1. - ind)) / (x_sorted[:, -1] - x_sorted[:, -3] + 1e-12)

    #
    def dlr_loss(self, x, y):
        """
        DLR loss with binary-safe fallback.
        - C >= 3 (original AutoAttack):  -(z_y - z_k) / (z_(1) - z_(3) + eps)
        - C = 2 (binary):                -(z_y - z_k) / (z_max - z_min + eps)
        Returns: [B] tensor
        """
        x_sorted, ind_sorted = x.sort(dim=1)     # ascending: [:, -1]=max=z_(1), [:, -2]=z_(2), [:, -3]=z_(3)
        B, C = x.shape
        u = torch.arange(B, device=x.device)

        if C >= 3:
            # --- ORIGINAL (unchanged) ---
            ind = (ind_sorted[:, -1] == y).float()  # is y the argmax?
            return -(x[u, y] - x_sorted[:, -2] * ind - x_sorted[:, -1] * (1. - ind)) / (x_sorted[:, -1] - x_sorted[:, -3] + 1e-12)

        elif C == 2:
            # --- BINARY-SAFE --- aligned with paper
            # y must be 0/1; pick the other class logit directly
            other = y ^ 1  # same as (1 - y), but explicit for {0,1}
            zy = x[u, y]
            zk = x[u, other]
            return -(zy - zk)  
            
        else:
            # C == 1 (degenerate): no meaningful margin
            return torch.zeros(B, device=x.device, dtype=x.dtype)


    def attack_single_run(self, x, y, x_init=None):
        if len(x.shape) < self.ndims:
            x = x.unsqueeze(0)
            y = y.unsqueeze(0)

        if self.norm == 'Linf':
            t = 2 * torch.rand(x.shape).to(self.device).detach() - 1
            x_adv = x + self.eps * torch.ones_like(x
                ).detach() * self.normalize(t)
        elif self.norm == 'L2':
            t = torch.randn(x.shape).to(self.device).detach()
            #x_adv = x + self.eps * torch.ones_like(x
            #   ).detach() * self.normalize(t)
            x_adv = x
        elif self.norm == 'L1':
            t = torch.randn(x.shape).to(self.device).detach()
            delta = L1_projection(x, t, self.eps)
            x_adv = x + t + delta
            
        
        
        
        
        if not x_init is None:
            x_adv = x_init.clone()
            if self.norm == 'L1' and self.verbose:
                print('[custom init] L1 perturbation {:.5f}'.format(
                    (x_adv - x).abs().view(x.shape[0], -1).sum(1).max()))
            
        
        x_adv = x_adv.clamp(0., 1.)
        x_best = x_adv.clone()
        x_best_adv = x_adv.clone()
        loss_steps = torch.zeros([self.n_iter, x.shape[0]]
            ).to(self.device)
        loss_best_steps = torch.zeros([self.n_iter + 1, x.shape[0]]
            ).to(self.device)
        acc_steps = torch.zeros_like(loss_best_steps)

        if not self.is_tf_model:
            if self.loss == 'ce':
                criterion_indiv = nn.CrossEntropyLoss(reduction='none')
            elif self.loss == 'ce-targeted-cfts':
                criterion_indiv = lambda x, y: -1. * F.cross_entropy(x, y,
                    reduction='none')
            elif self.loss == 'dlr':
                criterion_indiv = self.dlr_loss
            elif self.loss == 'dlr-targeted':
                criterion_indiv = self.dlr_loss_targeted
            elif self.loss == 'ce-targeted':
                criterion_indiv = self.ce_loss_targeted
            else:
                raise ValueError('unknowkn loss')
        else:
            if self.loss == 'ce':
                criterion_indiv = self.model.get_logits_loss_grad_xent
            elif self.loss == 'dlr':
                criterion_indiv = self.model.get_logits_loss_grad_dlr
            elif self.loss == 'dlr-targeted':
                criterion_indiv = self.model.get_logits_loss_grad_target
            else:
                raise ValueError('unknowkn loss')
        
        
        x_adv.requires_grad_()
        grad = torch.zeros_like(x)
        for _ in range(self.eot_iter):
            if not self.is_tf_model:
                with torch.enable_grad():
                    logits = self.model(x_adv)
                    loss_indiv = criterion_indiv(logits, y)
                    loss = loss_indiv.sum()

                grad += torch.autograd.grad(loss, [x_adv])[0].detach()
            else:
                if self.y_target is None:
                    logits, loss_indiv, grad_curr = criterion_indiv(x_adv, y)
                else:
                    logits, loss_indiv, grad_curr = criterion_indiv(x_adv, y,
                        self.y_target)
                grad += grad_curr
        
        grad /= float(self.eot_iter)
        grad_best = grad.clone()

        if self.loss in ['dlr', 'dlr-targeted']:
            # check if there are zero gradients
            check_zero_gradients(grad, logger=self.logger)
        
        acc = logits.detach().max(1)[1] == y
        acc_steps[0] = acc + 0
        loss_best = loss_indiv.detach().clone()

        alpha = 2. if self.norm in ['Linf', 'L2'] else 1. if self.norm in ['L1'] else 2e-2
        step_size = alpha * self.eps * torch.ones([x.shape[0], *(
            [1] * self.ndims)]).to(self.device).detach()
        x_adv_old = x_adv.clone()
        counter = 0
        k = self.n_iter_2 + 0
        n_fts = math.prod(self.orig_dim)
        if self.norm == 'L1':
            k = max(int(.04 * self.n_iter), 1)
            if x_init is None:
                topk = .2 * torch.ones([x.shape[0]], device=self.device)
                sp_old =  n_fts * torch.ones_like(topk)
            else:
                topk = L0_norm(x_adv - x) / n_fts / 1.5
                sp_old = L0_norm(x_adv - x)
            #print(topk[0], sp_old[0])
            adasp_redstep = 1.5
            adasp_minstep = 10.
            #print(step_size[0].item())
        counter3 = 0

        loss_best_last_check = loss_best.clone()
        reduced_last_check = torch.ones_like(loss_best)
        n_reduced = 0

        u = torch.arange(x.shape[0], device=self.device)
        for i in range(self.n_iter):
            ### gradient step
            with torch.no_grad():
                x_adv = x_adv.detach()
                grad2 = x_adv - x_adv_old
                x_adv_old = x_adv.clone()

                a = 0.75 if i > 0 else 1.0

                if self.norm == 'Linf':
                    x_adv_1 = x_adv + step_size * torch.sign(grad)
                    x_adv_1 = torch.clamp(torch.min(torch.max(x_adv_1,
                        x - self.eps), x + self.eps), 0.0, 1.0)
                    x_adv_1 = torch.clamp(torch.min(torch.max(
                        x_adv + (x_adv_1 - x_adv) * a + grad2 * (1 - a),
                        x - self.eps), x + self.eps), 0.0, 1.0)

                elif self.norm == 'L2':
                    x_adv_1 = x_adv + step_size * self.normalize(grad)
                    x_adv_1 = torch.clamp(x + self.normalize(x_adv_1 - x
                        ) * torch.min(self.eps * torch.ones_like(x).detach(),
                        L2_norm(x_adv_1 - x, keepdim=True)), 0.0, 1.0)
                    x_adv_1 = x_adv + (x_adv_1 - x_adv) * a + grad2 * (1 - a)
                    x_adv_1 = torch.clamp(x + self.normalize(x_adv_1 - x
                        ) * torch.min(self.eps * torch.ones_like(x).detach(),
                        L2_norm(x_adv_1 - x, keepdim=True)), 0.0, 1.0)

                elif self.norm == 'L1':
                    grad_topk = grad.abs().view(x.shape[0], -1).sort(-1)[0]
                    topk_curr = torch.clamp((1. - topk) * n_fts, min=0, max=n_fts - 1).long()
                    grad_topk = grad_topk[u, topk_curr].view(-1, *[1]*(len(x.shape) - 1))
                    sparsegrad = grad * (grad.abs() >= grad_topk).float()
                    x_adv_1 = x_adv + step_size * sparsegrad.sign() / (
                        L1_norm(sparsegrad.sign(), keepdim=True) + 1e-10)
                    
                    delta_u = x_adv_1 - x
                    delta_p = L1_projection(x, delta_u, self.eps)
                    x_adv_1 = x + delta_u + delta_p
                    
                    
                x_adv = x_adv_1 + 0.

            ### get gradient
            x_adv.requires_grad_()
            grad = torch.zeros_like(x)
            for _ in range(self.eot_iter):
                if not self.is_tf_model:
                    with torch.enable_grad():
                        logits = self.model(x_adv)
                        loss_indiv = criterion_indiv(logits, y)
                        loss = loss_indiv.sum()
    
                    grad += torch.autograd.grad(loss, [x_adv])[0].detach()
                else:
                    if self.y_target is None:
                        logits, loss_indiv, grad_curr = criterion_indiv(x_adv, y)
                    else:
                        logits, loss_indiv, grad_curr = criterion_indiv(x_adv, y, self.y_target)
                    grad += grad_curr
            
            grad /= float(self.eot_iter)

            pred = logits.detach().max(1)[1] == y
            acc = torch.min(acc, pred)
            acc_steps[i + 1] = acc + 0
            ind_pred = (pred == 0).nonzero().squeeze()
            x_best_adv[ind_pred] = x_adv[ind_pred] + 0.
            if self.verbose:
                str_stats = ' - step size: {:.5f} - topk: {:.2f}'.format(
                    step_size.mean(), topk.mean() * n_fts) if self.norm in ['L1'] else ''
                print('[m] iteration: {} - best loss: {:.6f} - robust accuracy: {:.2%}{}'.format(
                    i, loss_best.sum(), acc.float().mean(), str_stats))
                #print('pert {}'.format((x - x_best_adv).abs().view(x.shape[0], -1).sum(-1).max()))
            
            ### check step size
            with torch.no_grad():
              y1 = loss_indiv.detach().clone()
              loss_steps[i] = y1 + 0
              ind = (y1 > loss_best).nonzero().squeeze()
              x_best[ind] = x_adv[ind].clone()
              grad_best[ind] = grad[ind].clone()
              loss_best[ind] = y1[ind] + 0
              loss_best_steps[i + 1] = loss_best + 0

              counter3 += 1

              if counter3 == k:
                  if self.norm in ['Linf', 'L2']:
                      fl_oscillation = self.check_oscillation(loss_steps, i, k,
                          loss_best, k3=self.thr_decr)
                      fl_reduce_no_impr = (1. - reduced_last_check) * (
                          loss_best_last_check >= loss_best).float()
                      fl_oscillation = torch.max(fl_oscillation,
                          fl_reduce_no_impr)
                      reduced_last_check = fl_oscillation.clone()
                      loss_best_last_check = loss_best.clone()
    
                      if fl_oscillation.sum() > 0:
                          ind_fl_osc = (fl_oscillation > 0).nonzero().squeeze()
                          step_size[ind_fl_osc] /= 2.0
                          n_reduced = fl_oscillation.sum()
    
                          x_adv[ind_fl_osc] = x_best[ind_fl_osc].clone()
                          grad[ind_fl_osc] = grad_best[ind_fl_osc].clone()

                      k = max(k - self.size_decr, self.n_iter_min)
                  
                  elif self.norm == 'L1':
                      sp_curr = L0_norm(x_best - x)
                      fl_redtopk = (sp_curr / sp_old) < .95
                      topk = sp_curr / n_fts / 1.5
                      step_size[fl_redtopk] = alpha * self.eps
                      step_size[~fl_redtopk] /= adasp_redstep
                      step_size.clamp_(alpha * self.eps / adasp_minstep, alpha * self.eps)
                      sp_old = sp_curr.clone()
                  
                      x_adv[fl_redtopk] = x_best[fl_redtopk].clone()
                      grad[fl_redtopk] = grad_best[fl_redtopk].clone()
                  
                  counter3 = 0
                  #k = max(k - self.size_decr, self.n_iter_min)

        #
        
        return (x_best, acc, loss_best, x_best_adv)

        # Added by K to check the attack single run (no AutoAttack API)
    def APGDCroceAttackWrapper(self, device, dataLoader):
        """
        Runs self.attack_single_run() batch-by-batch on a DataLoader and returns
        a new DataLoader containing adversarial examples + original labels.

        Assumes DataManagerPytorch (DMP) is available.
        """
        self.device = device  # keep consistent with your notebook/device

        numSamples = len(dataLoader.dataset)
        xShape = DMP.GetOutputShape(dataLoader)   # expected (C,H,W)

        # store on CPU (safe for large tensors)
        xAdv   = torch.zeros((numSamples, *xShape), dtype=torch.float32)
        yClean = torch.zeros((numSamples,), dtype=torch.long)

        tracker = 0
        bs = dataLoader.batch_size if hasattr(dataLoader, "batch_size") else 32

        for xData, yData in dataLoader:
            batchSize = xData.shape[0]

            xData_d = xData.to(device)
            yData_d = yData.long().to(device)

            # run single APGD run
            xBest, acc, loss_best, x_best_adv = self.attack_single_run(
                xData_d, yData_d, x_init=None
            )

            # save back to CPU
            xAdv[tracker:tracker + batchSize]   = xBest.detach().cpu()
            yClean[tracker:tracker + batchSize] = yData.long().cpu()

            tracker += batchSize
            print(tracker, "/", numSamples, end="\r")

        advLoader = DMP.TensorToDataLoader(
            xAdv, yClean,
            transforms=None,
            batchSize=bs,
            randomizer=None
        )
        print()  # newline after progress
        return advLoader

    
    def perturb(self, x, y=None, best_loss=False, x_init=None):
        """
        :param x:           clean images
        :param y:           clean labels, if None we use the predicted labels
        :param best_loss:   if True the points attaining highest loss
                            are returned, otherwise adversarial examples
        """

        assert self.loss in ['ce', 'dlr'] #'ce-targeted-cfts'
        if not y is None and len(y.shape) == 0:
            x.unsqueeze_(0)
            y.unsqueeze_(0)
        self.init_hyperparam(x)

        x = x.detach().clone().float().to(self.device)
        if not self.is_tf_model:
            y_pred = self.model(x).max(1)[1]
        else:
            y_pred = self.model.predict(x).max(1)[1]
        if y is None:
            #y_pred = self.predict(x).max(1)[1]
            y = y_pred.detach().clone().long().to(self.device)
        else:
            y = y.detach().clone().long().to(self.device)

        adv = x.clone()
        if self.loss != 'ce-targeted':
            acc = y_pred == y
        else:
            acc = y_pred != y
        loss = -1e10 * torch.ones_like(acc).float()
        if self.verbose:
            print('-------------------------- ',
                'running {}-attack with epsilon {:.5f}'.format(
                self.norm, self.eps),
                '--------------------------')
            print('initial accuracy: {:.2%}'.format(acc.float().mean()))

        
        
        if self.use_largereps:
            epss = [3. * self.eps_orig, 2. * self.eps_orig, 1. * self.eps_orig]
            iters = [.3 * self.n_iter_orig, .3 * self.n_iter_orig,
                .4 * self.n_iter_orig]
            iters = [math.ceil(c) for c in iters]
            iters[-1] = self.n_iter_orig - sum(iters[:-1]) # make sure to use the given iterations
            if self.verbose:
                print('using schedule [{}x{}]'.format('+'.join([str(c
                    ) for c in epss]), '+'.join([str(c) for c in iters])))
        
        startt = time.time()
        if not best_loss:
            torch.random.manual_seed(self.seed)
            torch.cuda.random.manual_seed(self.seed)

            for counter in range(self.n_restarts):
                ind_to_fool = acc.nonzero().squeeze()
                if len(ind_to_fool.shape) == 0:
                    ind_to_fool = ind_to_fool.unsqueeze(0)
                if ind_to_fool.numel() != 0:
                    x_to_fool = x[ind_to_fool].clone()
                    y_to_fool = y[ind_to_fool].clone()
                    
                    
                    if not self.use_largereps:
                        res_curr = self.attack_single_run(x_to_fool, y_to_fool)
                    else:
                        res_curr = self.decr_eps_pgd(x_to_fool, y_to_fool, epss, iters)
                    best_curr, acc_curr, loss_curr, adv_curr = res_curr
                    ind_curr = (acc_curr == 0).nonzero().squeeze()

                    acc[ind_to_fool[ind_curr]] = 0
                    adv[ind_to_fool[ind_curr]] = adv_curr[ind_curr].clone()
                    if self.verbose:
                        print('restart {} - robust accuracy: {:.2%}'.format(
                            counter, acc.float().mean()),
                            '- cum. time: {:.1f} s'.format(
                            time.time() - startt))

            return adv

        else:
            adv_best = x.detach().clone()
            loss_best = torch.ones([x.shape[0]]).to(
                self.device) * (-float('inf'))
            for counter in range(self.n_restarts):
                best_curr, _, loss_curr, _ = self.attack_single_run(x, y)
                ind_curr = (loss_curr > loss_best).nonzero().squeeze()
                adv_best[ind_curr] = best_curr[ind_curr] + 0.
                loss_best[ind_curr] = loss_curr[ind_curr] + 0.

                if self.verbose:
                    print('restart {} - loss: {:.5f}'.format(
                        counter, loss_best.sum()))

            return adv_best

    def decr_eps_pgd(self, x, y, epss, iters, use_rs=True):
        assert len(epss) == len(iters)
        assert self.norm in ['L1']
        self.use_rs = False
        if not use_rs:
            x_init = None
        else:
            x_init = x + torch.randn_like(x)
            x_init += L1_projection(x, x_init - x, 1. * float(epss[0]))
        eps_target = float(epss[-1])
        if self.verbose:
            print('total iter: {}'.format(sum(iters)))
        for eps, niter in zip(epss, iters):
            if self.verbose:
                print('using eps: {:.2f}'.format(eps))
            self.n_iter = niter + 0
            self.eps = eps + 0.
            #
            if not x_init is None:
                x_init += L1_projection(x, x_init - x, 1. * eps)
            x_init, acc, loss, x_adv = self.attack_single_run(x, y, x_init=x_init)

        return (x_init, acc, loss, x_adv)

class APGDAttack_targeted(APGDAttack):
    def __init__(
            self,
            predict,
            n_iter=100,
            norm='Linf',
            n_restarts=1,
            eps=None,
            seed=0,
            eot_iter=1,
            rho=.75,
            topk=None,
            n_target_classes=9,
            verbose=False,
            device=None,
            use_largereps=False,
            is_tf_model=False,
            logger=None):
        """
        AutoPGD on the targeted DLR loss
        """
        super(APGDAttack_targeted, self).__init__(predict, n_iter=n_iter, norm=norm,
            n_restarts=n_restarts, eps=eps, seed=seed, loss='dlr-targeted',
            eot_iter=eot_iter, rho=rho, topk=topk, verbose=verbose, device=device,
            use_largereps=use_largereps, is_tf_model=is_tf_model, logger=logger)

        self.y_target = None
        self.n_target_classes = n_target_classes

    def dlr_loss_targeted(self, x, y):
        x_sorted, ind_sorted = x.sort(dim=1)
        u = torch.arange(x.shape[0])

        return -(x[u, y] - x[u, self.y_target]) / (x_sorted[:, -1] - .5 * (
            x_sorted[:, -3] + x_sorted[:, -4]) + 1e-12)

    def ce_loss_targeted(self, x, y):
        return -1. * F.cross_entropy(x, self.y_target, reduction='none')
    
    
    def perturb(self, x, y=None, x_init=None):
        """
        :param x:           clean images
        :param y:           clean labels, if None we use the predicted labels
        """

        assert self.loss in ['dlr-targeted'] #'ce-targeted'
        if not y is None and len(y.shape) == 0:
            x.unsqueeze_(0)
            y.unsqueeze_(0)
        self.init_hyperparam(x)

        x = x.detach().clone().float().to(self.device)
        if not self.is_tf_model:
            y_pred = self.model(x).max(1)[1]
        else:
            y_pred = self.model.predict(x).max(1)[1]
        if y is None:
            #y_pred = self._get_predicted_label(x)
            y = y_pred.detach().clone().long().to(self.device)
        else:
            y = y.detach().clone().long().to(self.device)

        adv = x.clone()
        acc = y_pred == y
        if self.verbose:
            print('-------------------------- ',
                'running {}-attack with epsilon {:.5f}'.format(
                self.norm, self.eps),
                '--------------------------')
            print('initial accuracy: {:.2%}'.format(acc.float().mean()))

        startt = time.time()

        torch.random.manual_seed(self.seed)
        torch.cuda.random.manual_seed(self.seed)

        #
        
        if self.use_largereps:
            epss = [3. * self.eps_orig, 2. * self.eps_orig, 1. * self.eps_orig]
            iters = [.3 * self.n_iter_orig, .3 * self.n_iter_orig,
                .4 * self.n_iter_orig]
            iters = [math.ceil(c) for c in iters]
            iters[-1] = self.n_iter_orig - sum(iters[:-1])
            if self.verbose:
                print('using schedule [{}x{}]'.format('+'.join([str(c
                    ) for c in epss]), '+'.join([str(c) for c in iters])))
        
        for target_class in range(2, self.n_target_classes + 2):
            for counter in range(self.n_restarts):
                ind_to_fool = acc.nonzero().squeeze()
                if len(ind_to_fool.shape) == 0:
                    ind_to_fool = ind_to_fool.unsqueeze(0)
                if ind_to_fool.numel() != 0:
                    x_to_fool = x[ind_to_fool].clone()
                    y_to_fool = y[ind_to_fool].clone()
                    
                    if not self.is_tf_model:
                        output = self.model(x_to_fool)
                    else:
                        output = self.model.predict(x_to_fool)
                    self.y_target = output.sort(dim=1)[1][:, -target_class]

                    if not self.use_largereps:
                        res_curr = self.attack_single_run(x_to_fool, y_to_fool)
                    else:
                        res_curr = self.decr_eps_pgd(x_to_fool, y_to_fool, epss, iters)
                    best_curr, acc_curr, loss_curr, adv_curr = res_curr
                    ind_curr = (acc_curr == 0).nonzero().squeeze()

                    acc[ind_to_fool[ind_curr]] = 0
                    adv[ind_to_fool[ind_curr]] = adv_curr[ind_curr].clone()
                    if self.verbose:
                        print('target class {}'.format(target_class),
                            '- restart {} - robust accuracy: {:.2%}'.format(
                            counter, acc.float().mean()),
                            '- cum. time: {:.1f} s'.format(
                            time.time() - startt))

        return adv
</file>

<file path="attacks/l2_attack/checks.py">
import torch
import warnings
import math
import sys

from .other_utils import L2_norm


funcs = {'grad': 0,
    'backward': 0,
    #'enable_grad': 0
    '_make_grads': 0,
    }

checks_doc_path = 'flags_doc.md'


def check_randomized(model, x, y, bs=250, n=5, alpha=1e-4, logger=None):
    acc = []
    corrcl = []
    outputs = []
    with torch.no_grad():
        for _ in range(n):
            output = model(x)
            corrcl_curr = (output.max(1)[1] == y).sum().item()
            corrcl.append(corrcl_curr)
            outputs.append(output / (L2_norm(output, keepdim=True) + 1e-10))
    acc = [c != corrcl_curr for c in corrcl]
    max_diff = 0.
    for c in range(n - 1):
        for e in range(c + 1, n):
            diff = L2_norm(outputs[c] - outputs[e])
            max_diff = max(max_diff, diff.max().item())
            #print(diff.max().item(), max_diff)
    if any(acc) or max_diff > alpha:
        msg = 'it seems to be a randomized defense! Please use version="rand".' + \
            f' See {checks_doc_path} for details.'
        if logger is None:
            warnings.warn(Warning(msg))
        else:
            logger.log(f'Warning: {msg}')


def check_range_output(model, x, alpha=1e-5, logger=None):
    with torch.no_grad():
        output = model(x)
    fl = [output.max() < 1. + alpha, output.min() >  -alpha,
        ((output.sum(-1) - 1.).abs() < alpha).all()]
    if all(fl):
        msg = 'it seems that the output is a probability distribution,' +\
            ' please be sure that the logits are used!' + \
            f' See {checks_doc_path} for details.'
        if logger is None:
            warnings.warn(Warning(msg))
        else:
            logger.log(f'Warning: {msg}')
    return output.shape[-1]


def check_zero_gradients(grad, logger=None):
    z = grad.view(grad.shape[0], -1).abs().sum(-1)
    #print(grad[0, :10])
    if (z == 0).any():
        msg = f'there are {(z == 0).sum()} points with zero gradient!' + \
            ' This might lead to unreliable evaluation with gradient-based attacks.' + \
            f' See {checks_doc_path} for details.'
        if logger is None:
            warnings.warn(Warning(msg))
        else:
            logger.log(f'Warning: {msg}')


def check_square_sr(acc_dict, alpha=.002, logger=None):
    if 'square' in acc_dict.keys() and len(acc_dict) > 2:
        acc = min([v for k, v in acc_dict.items() if k != 'square'])
        if acc_dict['square'] < acc - alpha:
            msg = 'Square Attack has decreased the robust accuracy of' + \
                f' {acc - acc_dict["square"]:.2%}.' + \
                ' This might indicate that the robustness evaluation using' +\
                ' AutoAttack is unreliable. Consider running Square' +\
                ' Attack with more iterations and restarts or an adaptive attack.' + \
                f' See {checks_doc_path} for details.'
            if logger is None:
                warnings.warn(Warning(msg))
            else:
                logger.log(f'Warning: {msg}')


''' from https://stackoverflow.com/questions/26119521/counting-function-calls-python '''
def tracefunc(frame, event, args):
    if event == 'call' and frame.f_code.co_name in funcs.keys():
        funcs[frame.f_code.co_name] += 1

        
def check_dynamic(model, x, is_tf_model=False, logger=None):
    if is_tf_model:
        msg = 'the check for dynamic defenses is not currently supported'
    else:
        msg = None
        sys.settrace(tracefunc)
        model(x)
        sys.settrace(None)
        #for k, v in funcs.items():
        #    print(k, v)
        if any([c > 0 for c in funcs.values()]):
            msg = 'it seems to be a dynamic defense! The evaluation' + \
                ' with AutoAttack might be insufficient.' + \
                f' See {checks_doc_path} for details.'
    if not msg is None:
        if logger is None:
            warnings.warn(Warning(msg))
        else:
            logger.log(f'Warning: {msg}')
    #sys.settrace(None)


def check_n_classes(n_cls, attacks_to_run, apgd_targets, fab_targets,
    logger=None):
    msg = None
    if 'apgd-dlr' in attacks_to_run or 'apgd-t' in attacks_to_run:
        if n_cls <= 2:
            msg = f'with only {n_cls} classes it is not possible to use the DLR loss!'
        elif n_cls == 3:
            msg = f'with only {n_cls} classes it is not possible to use the targeted DLR loss!'
        elif 'apgd-t' in attacks_to_run and \
            apgd_targets + 1 > n_cls:
            msg = f'it seems that more target classes ({apgd_targets})' + \
                f' than possible ({n_cls - 1}) are used in {"apgd-t".upper()}!'
    if 'fab-t' in attacks_to_run and fab_targets + 1 > n_cls:
        if msg is None:
            msg = f'it seems that more target classes ({apgd_targets})' + \
                f' than possible ({n_cls - 1}) are used in FAB-T!'
        else:
            msg += f' Also, it seems that too many target classes ({apgd_targets})' + \
                f' are used in {"fab-t".upper()} ({n_cls - 1} possible)!'
    if not msg is None:
        if logger is None:
            warnings.warn(Warning(msg))
        else:
            logger.log(f'Warning: {msg}')
</file>

<file path="attacks/l2_attack/DataManagerPytorch.py">
#DataManagerPytorch 
#Current Version Number = 1.1 (July 15, 2022), Please do not remove this comment.
import torch 
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import math 
from random import shuffle

#Convert an image dataloader (I) to a repeat encoding dataloader (E)
def DataLoaderIToDataLoaderRE(dataLoaderI, length):
    #First convert the image dataloader to tensor form
    xTensor, yTensor = DataLoaderToTensor(dataLoaderI)
    #Create memory for the new tensor with repeat encoding 
    xTensorRepeat = torch.zeros(xTensor.shape + (length,))
    #Go through and fill in the new array, probably a faster way to do this with Pytorch tensors
    for i in range(0, xTensor.shape[0]):
        for j in range(0, length):
            xTensorRepeat[i, :, :, :, j] = xTensor[i]
    #New tensor is filled in, convert back to dataloader
    dataLoaderRE = TensorToDataLoader(xTensorRepeat, yTensor, transforms=None, batchSize =dataLoaderI.batch_size, randomizer = None)
    return dataLoaderRE

#Convert an image dataloader (I) to a repeat encoding dataloader (E)
def DataLoaderREToDataLoaderI(dataLoaderRE):
    #First convert the repeated dataloader to tensor form
    xTensor, yTensor = DataLoaderToTensor(dataLoaderRE)
    #Create memory for the new tensor with repeat encoding 
    xTensorImages = torch.zeros(xTensor.shape[0], xTensor.shape[1], xTensor.shape[2], xTensor.shape[3])
    #Go through and fill in the new array, probably a faster way to do this with Pytorch tensors
    for i in range(0, xTensor.shape[0]):
        xTensorImages[i] = xTensor[i, :, :, :, 0] #Just take the first image from the repeated tensor because they should be the same
    #New tensor is filled in, convert back to dataloader
    dataLoaderI = TensorToDataLoader(xTensorImages, yTensor, transforms=None, batchSize =dataLoaderRE.batch_size, randomizer = None)
    return dataLoaderI

def CheckCudaMem():
    r = torch.cuda.memory_reserved(0) 
    a = torch.cuda.memory_allocated(0)
    f = r-a  # free inside reserved
    print("Unfree Memory=", a)

#Class to help with converting between dataloader and pytorch tensor 
class MyDataSet(torch.utils.data.Dataset):
    def __init__(self, x_tensor, y_tensor, transforms=None):
        self.x = x_tensor
        self.y = y_tensor
        self.transforms = transforms

    def __getitem__(self, index):
        if self.transforms is None: #No transform so return the data directly
            return (self.x[index], self.y[index])
        else: #Transform so apply it to the data before returning 
            return (self.transforms(self.x[index]), self.y[index])

    def __len__(self):
        return len(self.x)

#Validate using a dataloader 
def validateD(valLoader, model, device=None):
    #switch to evaluate mode
    model.eval()
    acc = 0 
    batchTracker = 0
    with torch.no_grad():
        #Go through and process the data in batches 
        for i, (input, target) in enumerate(valLoader):
            sampleSize = input.shape[0] #Get the number of samples used in each batch
            batchTracker = batchTracker + sampleSize
            #print("Processing up to sample=", batchTracker)
            if device == None: #assume cuda
                inputVar = input.cuda()
            else:
                inputVar = input.to(device)
            #compute output
            output = model(inputVar)
            output = output.float()
            #Go through and check how many samples correctly identified
            for j in range(0, sampleSize):
                if output[j].argmax(axis=0) == target[j]:
                    acc = acc +1
    acc = acc / float(len(valLoader.dataset))
    return acc

#Method to validate data using Pytorch tensor inputs and a Pytorch model 
def validateT(xData, yData, model, batchSize=None):
    acc = 0 #validation accuracy 
    numSamples = xData.shape[0]
    model.eval() #change to eval mode
    if batchSize == None: #No batch size so we can feed everything into the GPU
         output = model(xData)
         for i in range(0, numSamples):
             if output[i].argmax(axis=0) == yData[i]:
                 acc = acc+ 1
    else: #There are too many samples so we must process in batch
        numBatches = int(math.ceil(xData.shape[0] / batchSize)) #get the number of batches and type cast to int
        for i in range(0, numBatches): #Go through each batch 
            print(i)
            modelOutputIndex = 0 #reset output index
            startIndex = i*batchSize
            #change the end index depending on whether we are on the last batch or not:
            if i == numBatches-1: #last batch so go to the end
                endIndex = numSamples
            else: #Not the last batch so index normally
                endIndex = (i+1)*batchSize
            output = model(xData[startIndex:endIndex])
            for j in range(startIndex, endIndex): #check how many samples in the batch match the target
                if output[modelOutputIndex].argmax(axis=0) == yData[j]:
                    acc = acc+ 1
                modelOutputIndex = modelOutputIndex + 1 #update the output index regardless
    #Do final averaging and return 
    acc = acc / numSamples
    return acc

#Input a dataloader and model
#Instead of returning a model, output is array with 1.0 dentoting the sample was correctly identified
def validateDA(valLoader, model, device=None):
    numSamples = len(valLoader.dataset)
    accuracyArray = torch.zeros(numSamples) #variable for keep tracking of the correctly identified samples 
    #switch to evaluate mode
    model.eval()
    indexer = 0
    accuracy = 0
    batchTracker = 0
    with torch.no_grad():
        #Go through and process the data in batches 
        for i, (input, target) in enumerate(valLoader):
            sampleSize = input.shape[0] #Get the number of samples used in each batch
            batchTracker = batchTracker + sampleSize
            #print("Processing up to sample=", batchTracker)
            if device == None: #assume CUDA by default
                inputVar = input.cpu() #.cuda()
            else:
                inputVar = input.to(device) #use the prefered device if one is specified
            #compute output
            output = model(inputVar)
            output = output.float()
            #Go through and check how many samples correctly identified
            for j in range(0, sampleSize):
                if output[j].argmax(axis=0) == target[j]:
                    accuracyArray[indexer] = 1.0 #Mark with a 1.0 if sample is correctly identified
                    accuracy = accuracy + 1
                indexer = indexer + 1 #update the indexer regardless 
    accuracy = accuracy/numSamples
    print("Accuracy:", accuracy)
    return accuracyArray

#Replicate TF's predict method behavior 
def predictD(dataLoader, numClasses, model, device=None):
    numSamples = len(dataLoader.dataset)
    yPred = torch.zeros(numSamples, numClasses)
    #switch to evaluate mode
    model.eval()
    indexer = 0
    batchTracker = 0
    with torch.no_grad():
        #Go through and process the data in batches 
        for i, (input, target) in enumerate(dataLoader):
            sampleSize = input.shape[0] #Get the number of samples used in each batch
            batchTracker = batchTracker + sampleSize
            #print("Processing up to sample=", batchTracker)
            if device == None:
                inputVar = input.cuda()
            else:
                inputVar = input.to(device)
            #compute output
            output = model(inputVar)
            output = output.float()
            for j in range(0, sampleSize):
                yPred[indexer] = output[j]
                indexer = indexer + 1 #update the indexer regardless 
    return yPred

#Convert a X and Y tensors into a dataloader
#Does not put any transforms with the data  
def TensorToDataLoader(xData, yData, transforms= None, batchSize=None, randomizer = None):
    if batchSize is None: #If no batch size put all the data through 
        batchSize = xData.shape[0]
    dataset = MyDataSet(xData, yData, transforms)
    if randomizer == None: #No randomizer
        dataLoader = torch.utils.data.DataLoader(dataset=dataset,  batch_size=batchSize, shuffle=False)
    else: #randomizer needed 
        train_sampler = torch.utils.data.RandomSampler(dataset)
        dataLoader = torch.utils.data.DataLoader(dataset=dataset,  batch_size=batchSize, sampler=train_sampler, shuffle=False)
    return dataLoader

#Convert a dataloader into x and y tensors 
def DataLoaderToTensor(dataLoader):
    #First check how many samples in the dataset
    numSamples = len(dataLoader.dataset) 
    sampleShape = GetOutputShape(dataLoader) #Get the output shape from the dataloader
    sampleIndex = 0
    #xData = torch.zeros(numSamples, sampleShape[0], sampleShape[1], sampleShape[2])
    xData = torch.zeros((numSamples,) + sampleShape) #Make it generic shape for non-image datasets
    yData = torch.zeros(numSamples)
    #Go through and process the data in batches 
    for i, (input, target) in enumerate(dataLoader):
        batchSize = input.shape[0] #Get the number of samples used in each batch
        #Save the samples from the batch in a separate tensor 
        for batchIndex in range(0, batchSize):
            xData[sampleIndex] = input[batchIndex]
            yData[sampleIndex] = target[batchIndex]
            sampleIndex = sampleIndex + 1 #increment the sample index 
    return xData, yData 

#Get the output shape from the dataloader
def GetOutputShape(dataLoader):
    for i, (input, target) in enumerate(dataLoader):
        return input[0].shape

#This method randomly creates fake labels for the attack 
#The fake target is guaranteed to not be the same as the original class label 
def GenerateTargetsLabelRandomly(yData, numClasses):
    fTargetLabels=torch.zeros(len(yData))
    for i in range(0, len(yData)):
        targetLabel=random.randint(0,numClasses-1)
        while targetLabel==yData[i]:#Target and true label should not be the same 
            targetLabel=random.randint(0,numClasses-1) #Keep flipping until a different label is achieved 
        fTargetLabels[i]=targetLabel
    return fTargetLabels

#Return the first n correctly classified examples from a model 
#Note examples may not be class balanced 
def GetFirstCorrectlyIdentifiedExamples(device, dataLoader, model, numSamples):
    #First check how many samples in the dataset
    numSamplesTotal = len(dataLoader.dataset) 
    sampleShape = GetOutputShape(dataLoader) #Get the output shape from the dataloader
    sampleIndex = 0
    #xClean = torch.zeros(numSamples, sampleShape[0], sampleShape[1], sampleShape[2])
    xClean = torch.zeros((numSamples,) + sampleShape)
    yClean = torch.zeros(numSamples)
    #switch to evaluate mode
    model.eval()
    acc = 0 
    with torch.no_grad():
        #Go through and process the data in batches 
        for i, (input, target) in enumerate(dataLoader):
            batchSize = input.shape[0] #Get the number of samples used in each batch
            inputVar = input.to(device)
            #compute output
            output = model(inputVar)
            output = output.float()
            #Go through and check how many samples correctly identified
            for j in range(0, batchSize):
                #Add the sample if it is correctly identified and we are not at the limit
                if output[j].argmax(axis=0) == target[j] and sampleIndex<numSamples: 
                    xClean[sampleIndex] = input[j]
                    yClean[sampleIndex] = target[j]
                    sampleIndex = sampleIndex+1
    #Done collecting samples, time to covert to dataloader 
    cleanLoader = TensorToDataLoader(xClean, yClean, transforms=None, batchSize = dataLoader.batch_size, randomizer = None)
    return cleanLoader

def GetCorrectlyIdentifiedSamplesBalanced(model, totalSamplesRequired, dataLoader, numClasses, device=None):
    sampleShape = GetOutputShape(dataLoader)
    xData, yData = DataLoaderToTensor(dataLoader)
    #Basic error checking 
    if totalSamplesRequired % numClasses != 0:
        raise ValueError("The total number of samples in not evenly divisable by the number of classes.")
    #Get the number of samples needed for each class
    numSamplesPerClass = int(totalSamplesRequired/numClasses) 
    correctlyClassifiedSamples = torch.zeros((numClasses, numSamplesPerClass, sampleShape[0], sampleShape[1], sampleShape[2]))
    sanityCounter = torch.zeros((numClasses))
    #yPred = model.predict(xData)
    yPred = predictD(dataLoader, numClasses, model, device)
    for i in range(0, xData.shape[0]): #Go through every sample 
        predictedClass = yPred[i].argmax(axis=0)
        trueClass = yData[i]#.argmax(axis=0) 
        currentSavedCount = int(sanityCounter[int(trueClass)]) #Check how may samples we previously saved from this class
        #If the network predicts the sample correctly and we haven't saved enough samples from this class yet then save it
        if predictedClass == trueClass and currentSavedCount<numSamplesPerClass:
            correctlyClassifiedSamples[int(trueClass), currentSavedCount] = xData[i] #Save the sample 
            sanityCounter[int(trueClass)] = sanityCounter[int(trueClass)] + 1 #Add one to the count of saved samples for this class
    #Now we have gone through the entire network, make sure we have enough samples
    for c in range(0, numClasses):
        if sanityCounter[c] != numSamplesPerClass:
            raise ValueError("The network does not have enough correctly predicted samples for this class.")
    #Assume we have enough samples now, restore in a properly shaped array 
    xCorrect = torch.zeros((totalSamplesRequired, xData.shape[1], xData.shape[2], xData.shape[3]))
    yCorrect = torch.zeros((totalSamplesRequired))
    currentIndex = 0 #indexing for the final array
    for c in range(0, numClasses): #Go through each class
        for j in range(0, numSamplesPerClass): #For each sample in the class store it 
            xCorrect[currentIndex] = correctlyClassifiedSamples[c,j]
            yCorrect[currentIndex] = c
            #yCorrect[currentIndex, c] = 1.0
            currentIndex = currentIndex + 1 
    #return xCorrect, yCorrect
    cleanDataLoader = TensorToDataLoader(xCorrect, yCorrect, transforms = None, batchSize = dataLoader.batch_size, randomizer = None)
    return cleanDataLoader

def GetCorrectlyIdentifiedSamplesBalancedDefense(defense, totalSamplesRequired, dataLoader, numClasses, device):
    sampleShape = GetOutputShape(dataLoader)
    xData, yData = DataLoaderToTensor(dataLoader)
    #Basic error checking 
    if totalSamplesRequired % numClasses != 0:
        raise ValueError("The total number of samples in not evenly divisable by the number of classes.")
    #Get the number of samples needed for each class
    numSamplesPerClass = int(totalSamplesRequired/numClasses) 
    #correctlyClassifiedSamples = torch.zeros((numClasses, numSamplesPerClass, sampleShape[0], sampleShape[1], sampleShape[2]))
    correctlyClassifiedSamples = torch.zeros(((numClasses,) + (numSamplesPerClass,) + sampleShape))
    sanityCounter = torch.zeros((numClasses))
    #yPred = model.predict(xData)
    yPred = defense.predictD(dataLoader, numClasses, device)
    for i in range(0, xData.shape[0]): #Go through every sample 
        predictedClass = yPred[i].argmax(axis=0)
        trueClass = yData[i]#.argmax(axis=0) 
        currentSavedCount = int(sanityCounter[int(trueClass)]) #Check how may samples we previously saved from this class
        #If the network predicts the sample correctly and we haven't saved enough samples from this class yet then save it
        if predictedClass == trueClass and currentSavedCount<numSamplesPerClass:
            correctlyClassifiedSamples[int(trueClass), currentSavedCount] = xData[i] #Save the sample 
            sanityCounter[int(trueClass)] = sanityCounter[int(trueClass)] + 1 #Add one to the count of saved samples for this class
    #Now we have gone through the entire network, make sure we have enough samples
    for c in range(0, numClasses):
        if sanityCounter[c] != numSamplesPerClass:
            raise ValueError("The network does not have enough correctly predicted samples for this class.")
    #Assume we have enough samples now, restore in a properly shaped array 
    #xCorrect = torch.zeros((totalSamplesRequired, xData.shape[1], xData.shape[2], xData.shape[3]))
    xCorrect = torch.zeros(((totalSamplesRequired,) + sampleShape))
    yCorrect = torch.zeros((totalSamplesRequired))
    currentIndex = 0 #indexing for the final array
    for c in range(0, numClasses): #Go through each class
        for j in range(0, numSamplesPerClass): #For each sample in the class store it 
            xCorrect[currentIndex] = correctlyClassifiedSamples[c,j]
            yCorrect[currentIndex] = c
            #yCorrect[currentIndex, c] = 1.0
            currentIndex = currentIndex + 1 
    #return xCorrect, yCorrect
    cleanDataLoader = TensorToDataLoader(xCorrect, yCorrect, transforms = None, batchSize = dataLoader.batch_size, randomizer = None)
    return cleanDataLoader

#Manually shuffle the data loader assuming no transformations
def ManuallyShuffleDataLoader(dataLoader):
    xTest, yTest = DataLoaderToTensor(dataLoader)
    #Shuffle the indicies of the samples 
    indexList = []
    for i in range(0, xTest.shape[0]):
        indexList.append(i)
    shuffle(indexList)
    #Shuffle the samples and put them back in the dataloader 
    xTestShuffle = torch.zeros(xTest.shape)
    yTestShuffle = torch.zeros(yTest.shape)
    for i in range(0, xTest.shape[0]): 
        xTestShuffle[i] = xTest[indexList[i]]
        yTestShuffle[i] = yTest[indexList[i]]
    dataLoaderShuffled = TensorToDataLoader(xTestShuffle, yTestShuffle, transforms = None, batchSize = dataLoader.batch_size, randomizer = None)
    return dataLoaderShuffled
</file>

<file path="attacks/l2_attack/other_utils.py">
import os
import collections.abc as container_abcs

import torch

class Logger():
    def __init__(self, log_path):
        self.log_path = log_path
        
    def log(self, str_to_log):
        print(str_to_log)
        if not self.log_path is None:
            with open(self.log_path, 'a') as f:
                f.write(str_to_log + '\n')
                f.flush()
            
def check_imgs(adv, x, norm):
    delta = (adv - x).view(adv.shape[0], -1)
    if norm == 'Linf':
        res = delta.abs().max(dim=1)[0]
    elif norm == 'L2':
        res = (delta ** 2).sum(dim=1).sqrt()
    elif norm == 'L1':
        res = delta.abs().sum(dim=1)

    str_det = 'max {} pert: {:.5f}, nan in imgs: {}, max in imgs: {:.5f}, min in imgs: {:.5f}'.format(
        norm, res.max(), (adv != adv).sum(), adv.max(), adv.min())
    print(str_det)
    
    return str_det

def L1_norm(x, keepdim=False):
    z = x.abs().view(x.shape[0], -1).sum(-1)
    if keepdim:
        z = z.view(-1, *[1]*(len(x.shape) - 1))
    return z

def L2_norm(x, keepdim=False):
    z = (x ** 2).view(x.shape[0], -1).sum(-1).sqrt()
    if keepdim:
        z = z.view(-1, *[1]*(len(x.shape) - 1))
    return z

def L0_norm(x):
    return (x != 0.).view(x.shape[0], -1).sum(-1)

def makedir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def zero_gradients(x):
    if isinstance(x, torch.Tensor):
        if x.grad is not None:
            x.grad.detach_()
            x.grad.zero_()
    elif isinstance(x, container_abcs.Iterable):
        for elem in x:
            zero_gradients(elem)
</file>

<file path="attacks/l2_attack/repo.md">
This file is a merged representation of the entire codebase, combined into a single document by Repomix.

<file_summary>
This section contains a summary of this file.

<purpose>
This file contains a packed representation of the entire repository's contents.
It is designed to be easily consumable by AI systems for analysis, code review,
or other automated processes.
</purpose>

<file_format>
The content is organized as follows:
1. This summary section
2. Repository information
3. Directory structure
4. Repository files (if enabled)
5. Multiple file entries, each consisting of:
  - File path as an attribute
  - Full contents of the file
</file_format>

<usage_guidelines>
- This file should be treated as read-only. Any changes should be made to the
  original repository files, not this packed version.
- When processing this file, use the file path to distinguish
  between different files in the repository.
- Be aware that this file may contain sensitive information. Handle it with
  the same level of security as you would the original repository.
</usage_guidelines>

<notes>
- Some files may have been excluded based on .gitignore rules and Repomix's configuration
- Binary files are not included in this packed representation. Please refer to the Repository Structure section for a complete list of file paths, including binary files
- Files matching patterns in .gitignore are excluded
- Files matching default ignore patterns are excluded
- Files are sorted by Git change count (files with more changes are at the bottom)
</notes>

</file_summary>

<directory_structure>
autopgd_base.py
checks.py
DataManagerPytorch.py
other_utils.py
</directory_structure>

<files>
This section contains the contents of the repository's files.

<file path="autopgd_base.py">
# Copyright (c) 2020-present, Francesco Croce
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree
#

import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import random

from other_utils import L0_norm, L1_norm, L2_norm
from checks import check_zero_gradients

import DataManagerPytorch as DMP


def L1_projection(x2, y2, eps1):
    '''
    x2: center of the L1 ball (bs x input_dim)
    y2: current perturbation (x2 + y2 is the point to be projected)
    eps1: radius of the L1 ball

    output: delta s.th. ||y2 + delta||_1 <= eps1
    and 0 <= x2 + y2 + delta <= 1
    '''

    x = x2.clone().float().view(x2.shape[0], -1)
    y = y2.clone().float().view(y2.shape[0], -1)
    sigma = y.clone().sign()
    u = torch.min(1 - x - y, x + y)
    #u = torch.min(u, epsinf - torch.clone(y).abs())
    u = torch.min(torch.zeros_like(y), u)
    l = -torch.clone(y).abs()
    d = u.clone()
    
    bs, indbs = torch.sort(-torch.cat((u, l), 1), dim=1)
    bs2 = torch.cat((bs[:, 1:], torch.zeros(bs.shape[0], 1).to(bs.device)), 1)
    
    inu = 2*(indbs < u.shape[1]).float() - 1
    size1 = inu.cumsum(dim=1)
    
    s1 = -u.sum(dim=1)
    
    c = eps1 - y.clone().abs().sum(dim=1)
    c5 = s1 + c < 0
    c2 = c5.nonzero().squeeze(1)
    
    s = s1.unsqueeze(-1) + torch.cumsum((bs2 - bs) * size1, dim=1)
    
    if c2.nelement != 0:
    
      lb = torch.zeros_like(c2).float()
      ub = torch.ones_like(lb) *(bs.shape[1] - 1)
      
      #print(c2.shape, lb.shape)
      
      nitermax = torch.ceil(torch.log2(torch.tensor(bs.shape[1]).float()))
      counter2 = torch.zeros_like(lb).long()
      counter = 0
          
      while counter < nitermax:
        counter4 = torch.floor((lb + ub) / 2.)
        counter2 = counter4.type(torch.LongTensor)
        
        c8 = s[c2, counter2] + c[c2] < 0
        ind3 = c8.nonzero().squeeze(1)
        ind32 = (~c8).nonzero().squeeze(1)
        #print(ind3.shape)
        if ind3.nelement != 0:
            lb[ind3] = counter4[ind3]
        if ind32.nelement != 0:
            ub[ind32] = counter4[ind32]
        
        #print(lb, ub)
        counter += 1
        
      lb2 = lb.long()
      alpha = (-s[c2, lb2] -c[c2]) / size1[c2, lb2 + 1] + bs2[c2, lb2]
      d[c2] = -torch.min(torch.max(-u[c2], alpha.unsqueeze(-1)), -l[c2])
    
    return (sigma * d).view(x2.shape)





class APGDAttack():
    """
    AutoPGD
    https://arxiv.org/abs/2003.01690

    :param predict:       forward pass function
    :param norm:          Lp-norm of the attack ('Linf', 'L2', 'L0' supported)
    :param n_restarts:    number of random restarts
    :param n_iter:        number of iterations
    :param eps:           bound on the norm of perturbations
    :param seed:          random seed for the starting point
    :param loss:          loss to optimize ('ce', 'dlr' supported)
    :param eot_iter:      iterations for Expectation over Trasformation
    :param rho:           parameter for decreasing the step size
    """

    def __init__(
            self,
            predict,
            n_iter=100,
            norm='Linf',
            n_restarts=1,
            eps=None,
            seed=0,
            loss='ce',
            eot_iter=1,
            rho=.75,
            topk=None,
            verbose=False,
            device=None,
            use_largereps=False,
            is_tf_model=False,
            logger=None):
        """
        AutoPGD implementation in PyTorch
        """
        
        self.model = predict
        self.n_iter = n_iter
        self.eps = eps
        self.norm = norm
        self.n_restarts = n_restarts
        self.seed = seed
        self.loss = loss
        self.eot_iter = eot_iter
        self.thr_decr = rho
        self.topk = topk
        self.verbose = verbose
        self.device = device
        self.use_rs = True
        #self.init_point = None
        self.use_largereps = use_largereps
        #self.larger_epss = None
        #self.iters = None
        self.n_iter_orig = n_iter + 0
        self.eps_orig = eps + 0.
        self.is_tf_model = is_tf_model
        self.y_target = None
        self.logger = logger

        assert self.norm in ['Linf', 'L2', 'L1']
        assert not self.eps is None

        ### set parameters for checkpoints
        self.n_iter_2 = max(int(0.22 * self.n_iter), 1)
        self.n_iter_min = max(int(0.06 * self.n_iter), 1)
        self.size_decr = max(int(0.03 * self.n_iter), 1)

    def init_hyperparam(self, x):

        if self.device is None:
            self.device = x.device
        self.orig_dim = list(x.shape[1:])
        self.ndims = len(self.orig_dim)
        if self.seed is None:
            self.seed = time.time()

    def check_oscillation(self, x, j, k, y5, k3=0.75):
        t = torch.zeros(x.shape[1]).to(self.device)
        for counter5 in range(k):
          t += (x[j - counter5] > x[j - counter5 - 1]).float()

        return (t <= k * k3 * torch.ones_like(t)).float()

    def check_shape(self, x):
        return x if len(x.shape) > 0 else x.unsqueeze(0)

    def normalize(self, x):
        if self.norm == 'Linf':
            t = x.abs().view(x.shape[0], -1).max(1)[0]

        elif self.norm == 'L2':
            t = (x ** 2).view(x.shape[0], -1).sum(-1).sqrt()

        elif self.norm == 'L1':
            try:
                t = x.abs().view(x.shape[0], -1).sum(dim=-1)
            except:
                t = x.abs().reshape([x.shape[0], -1]).sum(dim=-1)

        return x / (t.view(-1, *([1] * self.ndims)) + 1e-12)

    #def dlr_loss(self, x, y):
    #    x_sorted, ind_sorted = x.sort(dim=1)
    #    ind = (ind_sorted[:, -1] == y).float()
    #    u = torch.arange(x.shape[0])

     #   return -(x[u, y] - x_sorted[:, -2] * ind - x_sorted[:, -1] * (
      #      1. - ind)) / (x_sorted[:, -1] - x_sorted[:, -3] + 1e-12)

    #
    def dlr_loss(self, x, y):
        """
        DLR loss with binary-safe fallback.
        - C >= 3 (original AutoAttack):  -(z_y - z_k) / (z_(1) - z_(3) + eps)
        - C = 2 (binary):                -(z_y - z_k) / (z_max - z_min + eps)
        Returns: [B] tensor
        """
        x_sorted, ind_sorted = x.sort(dim=1)     # ascending: [:, -1]=max=z_(1), [:, -2]=z_(2), [:, -3]=z_(3)
        B, C = x.shape
        u = torch.arange(B, device=x.device)

        if C >= 3:
            # --- ORIGINAL (unchanged) ---
            ind = (ind_sorted[:, -1] == y).float()  # is y the argmax?
            return -(x[u, y] - x_sorted[:, -2] * ind - x_sorted[:, -1] * (1. - ind)) / (x_sorted[:, -1] - x_sorted[:, -3] + 1e-12)

        elif C == 2:
            # --- BINARY-SAFE --- aligned with paper
            # y must be 0/1; pick the other class logit directly
            other = y ^ 1  # same as (1 - y), but explicit for {0,1}
            zy = x[u, y]
            zk = x[u, other]
            return -(zy - zk)  
            
        else:
            # C == 1 (degenerate): no meaningful margin
            return torch.zeros(B, device=x.device, dtype=x.dtype)


    def attack_single_run(self, x, y, x_init=None):
        if len(x.shape) < self.ndims:
            x = x.unsqueeze(0)
            y = y.unsqueeze(0)

        if self.norm == 'Linf':
            t = 2 * torch.rand(x.shape).to(self.device).detach() - 1
            x_adv = x + self.eps * torch.ones_like(x
                ).detach() * self.normalize(t)
        elif self.norm == 'L2':
            t = torch.randn(x.shape).to(self.device).detach()
            #x_adv = x + self.eps * torch.ones_like(x
            #   ).detach() * self.normalize(t)
            x_adv = x
        elif self.norm == 'L1':
            t = torch.randn(x.shape).to(self.device).detach()
            delta = L1_projection(x, t, self.eps)
            x_adv = x + t + delta
            
        
        
        
        
        if not x_init is None:
            x_adv = x_init.clone()
            if self.norm == 'L1' and self.verbose:
                print('[custom init] L1 perturbation {:.5f}'.format(
                    (x_adv - x).abs().view(x.shape[0], -1).sum(1).max()))
            
        
        x_adv = x_adv.clamp(0., 1.)
        x_best = x_adv.clone()
        x_best_adv = x_adv.clone()
        loss_steps = torch.zeros([self.n_iter, x.shape[0]]
            ).to(self.device)
        loss_best_steps = torch.zeros([self.n_iter + 1, x.shape[0]]
            ).to(self.device)
        acc_steps = torch.zeros_like(loss_best_steps)

        if not self.is_tf_model:
            if self.loss == 'ce':
                criterion_indiv = nn.CrossEntropyLoss(reduction='none')
            elif self.loss == 'ce-targeted-cfts':
                criterion_indiv = lambda x, y: -1. * F.cross_entropy(x, y,
                    reduction='none')
            elif self.loss == 'dlr':
                criterion_indiv = self.dlr_loss
            elif self.loss == 'dlr-targeted':
                criterion_indiv = self.dlr_loss_targeted
            elif self.loss == 'ce-targeted':
                criterion_indiv = self.ce_loss_targeted
            else:
                raise ValueError('unknowkn loss')
        else:
            if self.loss == 'ce':
                criterion_indiv = self.model.get_logits_loss_grad_xent
            elif self.loss == 'dlr':
                criterion_indiv = self.model.get_logits_loss_grad_dlr
            elif self.loss == 'dlr-targeted':
                criterion_indiv = self.model.get_logits_loss_grad_target
            else:
                raise ValueError('unknowkn loss')
        
        
        x_adv.requires_grad_()
        grad = torch.zeros_like(x)
        for _ in range(self.eot_iter):
            if not self.is_tf_model:
                with torch.enable_grad():
                    logits = self.model(x_adv)
                    loss_indiv = criterion_indiv(logits, y)
                    loss = loss_indiv.sum()

                grad += torch.autograd.grad(loss, [x_adv])[0].detach()
            else:
                if self.y_target is None:
                    logits, loss_indiv, grad_curr = criterion_indiv(x_adv, y)
                else:
                    logits, loss_indiv, grad_curr = criterion_indiv(x_adv, y,
                        self.y_target)
                grad += grad_curr
        
        grad /= float(self.eot_iter)
        grad_best = grad.clone()

        if self.loss in ['dlr', 'dlr-targeted']:
            # check if there are zero gradients
            check_zero_gradients(grad, logger=self.logger)
        
        acc = logits.detach().max(1)[1] == y
        acc_steps[0] = acc + 0
        loss_best = loss_indiv.detach().clone()

        alpha = 2. if self.norm in ['Linf', 'L2'] else 1. if self.norm in ['L1'] else 2e-2
        step_size = alpha * self.eps * torch.ones([x.shape[0], *(
            [1] * self.ndims)]).to(self.device).detach()
        x_adv_old = x_adv.clone()
        counter = 0
        k = self.n_iter_2 + 0
        n_fts = math.prod(self.orig_dim)
        if self.norm == 'L1':
            k = max(int(.04 * self.n_iter), 1)
            if x_init is None:
                topk = .2 * torch.ones([x.shape[0]], device=self.device)
                sp_old =  n_fts * torch.ones_like(topk)
            else:
                topk = L0_norm(x_adv - x) / n_fts / 1.5
                sp_old = L0_norm(x_adv - x)
            #print(topk[0], sp_old[0])
            adasp_redstep = 1.5
            adasp_minstep = 10.
            #print(step_size[0].item())
        counter3 = 0

        loss_best_last_check = loss_best.clone()
        reduced_last_check = torch.ones_like(loss_best)
        n_reduced = 0

        u = torch.arange(x.shape[0], device=self.device)
        for i in range(self.n_iter):
            ### gradient step
            with torch.no_grad():
                x_adv = x_adv.detach()
                grad2 = x_adv - x_adv_old
                x_adv_old = x_adv.clone()

                a = 0.75 if i > 0 else 1.0

                if self.norm == 'Linf':
                    x_adv_1 = x_adv + step_size * torch.sign(grad)
                    x_adv_1 = torch.clamp(torch.min(torch.max(x_adv_1,
                        x - self.eps), x + self.eps), 0.0, 1.0)
                    x_adv_1 = torch.clamp(torch.min(torch.max(
                        x_adv + (x_adv_1 - x_adv) * a + grad2 * (1 - a),
                        x - self.eps), x + self.eps), 0.0, 1.0)

                elif self.norm == 'L2':
                    x_adv_1 = x_adv + step_size * self.normalize(grad)
                    x_adv_1 = torch.clamp(x + self.normalize(x_adv_1 - x
                        ) * torch.min(self.eps * torch.ones_like(x).detach(),
                        L2_norm(x_adv_1 - x, keepdim=True)), 0.0, 1.0)
                    x_adv_1 = x_adv + (x_adv_1 - x_adv) * a + grad2 * (1 - a)
                    x_adv_1 = torch.clamp(x + self.normalize(x_adv_1 - x
                        ) * torch.min(self.eps * torch.ones_like(x).detach(),
                        L2_norm(x_adv_1 - x, keepdim=True)), 0.0, 1.0)

                elif self.norm == 'L1':
                    grad_topk = grad.abs().view(x.shape[0], -1).sort(-1)[0]
                    topk_curr = torch.clamp((1. - topk) * n_fts, min=0, max=n_fts - 1).long()
                    grad_topk = grad_topk[u, topk_curr].view(-1, *[1]*(len(x.shape) - 1))
                    sparsegrad = grad * (grad.abs() >= grad_topk).float()
                    x_adv_1 = x_adv + step_size * sparsegrad.sign() / (
                        L1_norm(sparsegrad.sign(), keepdim=True) + 1e-10)
                    
                    delta_u = x_adv_1 - x
                    delta_p = L1_projection(x, delta_u, self.eps)
                    x_adv_1 = x + delta_u + delta_p
                    
                    
                x_adv = x_adv_1 + 0.

            ### get gradient
            x_adv.requires_grad_()
            grad = torch.zeros_like(x)
            for _ in range(self.eot_iter):
                if not self.is_tf_model:
                    with torch.enable_grad():
                        logits = self.model(x_adv)
                        loss_indiv = criterion_indiv(logits, y)
                        loss = loss_indiv.sum()
    
                    grad += torch.autograd.grad(loss, [x_adv])[0].detach()
                else:
                    if self.y_target is None:
                        logits, loss_indiv, grad_curr = criterion_indiv(x_adv, y)
                    else:
                        logits, loss_indiv, grad_curr = criterion_indiv(x_adv, y, self.y_target)
                    grad += grad_curr
            
            grad /= float(self.eot_iter)

            pred = logits.detach().max(1)[1] == y
            acc = torch.min(acc, pred)
            acc_steps[i + 1] = acc + 0
            ind_pred = (pred == 0).nonzero().squeeze()
            x_best_adv[ind_pred] = x_adv[ind_pred] + 0.
            if self.verbose:
                str_stats = ' - step size: {:.5f} - topk: {:.2f}'.format(
                    step_size.mean(), topk.mean() * n_fts) if self.norm in ['L1'] else ''
                print('[m] iteration: {} - best loss: {:.6f} - robust accuracy: {:.2%}{}'.format(
                    i, loss_best.sum(), acc.float().mean(), str_stats))
                #print('pert {}'.format((x - x_best_adv).abs().view(x.shape[0], -1).sum(-1).max()))
            
            ### check step size
            with torch.no_grad():
              y1 = loss_indiv.detach().clone()
              loss_steps[i] = y1 + 0
              ind = (y1 > loss_best).nonzero().squeeze()
              x_best[ind] = x_adv[ind].clone()
              grad_best[ind] = grad[ind].clone()
              loss_best[ind] = y1[ind] + 0
              loss_best_steps[i + 1] = loss_best + 0

              counter3 += 1

              if counter3 == k:
                  if self.norm in ['Linf', 'L2']:
                      fl_oscillation = self.check_oscillation(loss_steps, i, k,
                          loss_best, k3=self.thr_decr)
                      fl_reduce_no_impr = (1. - reduced_last_check) * (
                          loss_best_last_check >= loss_best).float()
                      fl_oscillation = torch.max(fl_oscillation,
                          fl_reduce_no_impr)
                      reduced_last_check = fl_oscillation.clone()
                      loss_best_last_check = loss_best.clone()
    
                      if fl_oscillation.sum() > 0:
                          ind_fl_osc = (fl_oscillation > 0).nonzero().squeeze()
                          step_size[ind_fl_osc] /= 2.0
                          n_reduced = fl_oscillation.sum()
    
                          x_adv[ind_fl_osc] = x_best[ind_fl_osc].clone()
                          grad[ind_fl_osc] = grad_best[ind_fl_osc].clone()

                      k = max(k - self.size_decr, self.n_iter_min)
                  
                  elif self.norm == 'L1':
                      sp_curr = L0_norm(x_best - x)
                      fl_redtopk = (sp_curr / sp_old) < .95
                      topk = sp_curr / n_fts / 1.5
                      step_size[fl_redtopk] = alpha * self.eps
                      step_size[~fl_redtopk] /= adasp_redstep
                      step_size.clamp_(alpha * self.eps / adasp_minstep, alpha * self.eps)
                      sp_old = sp_curr.clone()
                  
                      x_adv[fl_redtopk] = x_best[fl_redtopk].clone()
                      grad[fl_redtopk] = grad_best[fl_redtopk].clone()
                  
                  counter3 = 0
                  #k = max(k - self.size_decr, self.n_iter_min)

        #
        
        return (x_best, acc, loss_best, x_best_adv)

        # Added by K to check the attack single run (no AutoAttack API)
    def APGDCroceAttackWrapper(self, device, dataLoader):
        """
        Runs self.attack_single_run() batch-by-batch on a DataLoader and returns
        a new DataLoader containing adversarial examples + original labels.

        Assumes DataManagerPytorch (DMP) is available.
        """
        self.device = device  # keep consistent with your notebook/device

        numSamples = len(dataLoader.dataset)
        xShape = DMP.GetOutputShape(dataLoader)   # expected (C,H,W)

        # store on CPU (safe for large tensors)
        xAdv   = torch.zeros((numSamples, *xShape), dtype=torch.float32)
        yClean = torch.zeros((numSamples,), dtype=torch.long)

        tracker = 0
        bs = dataLoader.batch_size if hasattr(dataLoader, "batch_size") else 32

        for xData, yData in dataLoader:
            batchSize = xData.shape[0]

            xData_d = xData.to(device)
            yData_d = yData.long().to(device)

            # run single APGD run
            xBest, acc, loss_best, x_best_adv = self.attack_single_run(
                xData_d, yData_d, x_init=None
            )

            # save back to CPU
            xAdv[tracker:tracker + batchSize]   = xBest.detach().cpu()
            yClean[tracker:tracker + batchSize] = yData.long().cpu()

            tracker += batchSize
            print(tracker, "/", numSamples, end="\r")

        advLoader = DMP.TensorToDataLoader(
            xAdv, yClean,
            transforms=None,
            batchSize=bs,
            randomizer=None
        )
        print()  # newline after progress
        return advLoader

    
    def perturb(self, x, y=None, best_loss=False, x_init=None):
        """
        :param x:           clean images
        :param y:           clean labels, if None we use the predicted labels
        :param best_loss:   if True the points attaining highest loss
                            are returned, otherwise adversarial examples
        """

        assert self.loss in ['ce', 'dlr'] #'ce-targeted-cfts'
        if not y is None and len(y.shape) == 0:
            x.unsqueeze_(0)
            y.unsqueeze_(0)
        self.init_hyperparam(x)

        x = x.detach().clone().float().to(self.device)
        if not self.is_tf_model:
            y_pred = self.model(x).max(1)[1]
        else:
            y_pred = self.model.predict(x).max(1)[1]
        if y is None:
            #y_pred = self.predict(x).max(1)[1]
            y = y_pred.detach().clone().long().to(self.device)
        else:
            y = y.detach().clone().long().to(self.device)

        adv = x.clone()
        if self.loss != 'ce-targeted':
            acc = y_pred == y
        else:
            acc = y_pred != y
        loss = -1e10 * torch.ones_like(acc).float()
        if self.verbose:
            print('-------------------------- ',
                'running {}-attack with epsilon {:.5f}'.format(
                self.norm, self.eps),
                '--------------------------')
            print('initial accuracy: {:.2%}'.format(acc.float().mean()))

        
        
        if self.use_largereps:
            epss = [3. * self.eps_orig, 2. * self.eps_orig, 1. * self.eps_orig]
            iters = [.3 * self.n_iter_orig, .3 * self.n_iter_orig,
                .4 * self.n_iter_orig]
            iters = [math.ceil(c) for c in iters]
            iters[-1] = self.n_iter_orig - sum(iters[:-1]) # make sure to use the given iterations
            if self.verbose:
                print('using schedule [{}x{}]'.format('+'.join([str(c
                    ) for c in epss]), '+'.join([str(c) for c in iters])))
        
        startt = time.time()
        if not best_loss:
            torch.random.manual_seed(self.seed)
            torch.cuda.random.manual_seed(self.seed)

            for counter in range(self.n_restarts):
                ind_to_fool = acc.nonzero().squeeze()
                if len(ind_to_fool.shape) == 0:
                    ind_to_fool = ind_to_fool.unsqueeze(0)
                if ind_to_fool.numel() != 0:
                    x_to_fool = x[ind_to_fool].clone()
                    y_to_fool = y[ind_to_fool].clone()
                    
                    
                    if not self.use_largereps:
                        res_curr = self.attack_single_run(x_to_fool, y_to_fool)
                    else:
                        res_curr = self.decr_eps_pgd(x_to_fool, y_to_fool, epss, iters)
                    best_curr, acc_curr, loss_curr, adv_curr = res_curr
                    ind_curr = (acc_curr == 0).nonzero().squeeze()

                    acc[ind_to_fool[ind_curr]] = 0
                    adv[ind_to_fool[ind_curr]] = adv_curr[ind_curr].clone()
                    if self.verbose:
                        print('restart {} - robust accuracy: {:.2%}'.format(
                            counter, acc.float().mean()),
                            '- cum. time: {:.1f} s'.format(
                            time.time() - startt))

            return adv

        else:
            adv_best = x.detach().clone()
            loss_best = torch.ones([x.shape[0]]).to(
                self.device) * (-float('inf'))
            for counter in range(self.n_restarts):
                best_curr, _, loss_curr, _ = self.attack_single_run(x, y)
                ind_curr = (loss_curr > loss_best).nonzero().squeeze()
                adv_best[ind_curr] = best_curr[ind_curr] + 0.
                loss_best[ind_curr] = loss_curr[ind_curr] + 0.

                if self.verbose:
                    print('restart {} - loss: {:.5f}'.format(
                        counter, loss_best.sum()))

            return adv_best

    def decr_eps_pgd(self, x, y, epss, iters, use_rs=True):
        assert len(epss) == len(iters)
        assert self.norm in ['L1']
        self.use_rs = False
        if not use_rs:
            x_init = None
        else:
            x_init = x + torch.randn_like(x)
            x_init += L1_projection(x, x_init - x, 1. * float(epss[0]))
        eps_target = float(epss[-1])
        if self.verbose:
            print('total iter: {}'.format(sum(iters)))
        for eps, niter in zip(epss, iters):
            if self.verbose:
                print('using eps: {:.2f}'.format(eps))
            self.n_iter = niter + 0
            self.eps = eps + 0.
            #
            if not x_init is None:
                x_init += L1_projection(x, x_init - x, 1. * eps)
            x_init, acc, loss, x_adv = self.attack_single_run(x, y, x_init=x_init)

        return (x_init, acc, loss, x_adv)

class APGDAttack_targeted(APGDAttack):
    def __init__(
            self,
            predict,
            n_iter=100,
            norm='Linf',
            n_restarts=1,
            eps=None,
            seed=0,
            eot_iter=1,
            rho=.75,
            topk=None,
            n_target_classes=9,
            verbose=False,
            device=None,
            use_largereps=False,
            is_tf_model=False,
            logger=None):
        """
        AutoPGD on the targeted DLR loss
        """
        super(APGDAttack_targeted, self).__init__(predict, n_iter=n_iter, norm=norm,
            n_restarts=n_restarts, eps=eps, seed=seed, loss='dlr-targeted',
            eot_iter=eot_iter, rho=rho, topk=topk, verbose=verbose, device=device,
            use_largereps=use_largereps, is_tf_model=is_tf_model, logger=logger)

        self.y_target = None
        self.n_target_classes = n_target_classes

    def dlr_loss_targeted(self, x, y):
        x_sorted, ind_sorted = x.sort(dim=1)
        u = torch.arange(x.shape[0])

        return -(x[u, y] - x[u, self.y_target]) / (x_sorted[:, -1] - .5 * (
            x_sorted[:, -3] + x_sorted[:, -4]) + 1e-12)

    def ce_loss_targeted(self, x, y):
        return -1. * F.cross_entropy(x, self.y_target, reduction='none')
    
    
    def perturb(self, x, y=None, x_init=None):
        """
        :param x:           clean images
        :param y:           clean labels, if None we use the predicted labels
        """

        assert self.loss in ['dlr-targeted'] #'ce-targeted'
        if not y is None and len(y.shape) == 0:
            x.unsqueeze_(0)
            y.unsqueeze_(0)
        self.init_hyperparam(x)

        x = x.detach().clone().float().to(self.device)
        if not self.is_tf_model:
            y_pred = self.model(x).max(1)[1]
        else:
            y_pred = self.model.predict(x).max(1)[1]
        if y is None:
            #y_pred = self._get_predicted_label(x)
            y = y_pred.detach().clone().long().to(self.device)
        else:
            y = y.detach().clone().long().to(self.device)

        adv = x.clone()
        acc = y_pred == y
        if self.verbose:
            print('-------------------------- ',
                'running {}-attack with epsilon {:.5f}'.format(
                self.norm, self.eps),
                '--------------------------')
            print('initial accuracy: {:.2%}'.format(acc.float().mean()))

        startt = time.time()

        torch.random.manual_seed(self.seed)
        torch.cuda.random.manual_seed(self.seed)

        #
        
        if self.use_largereps:
            epss = [3. * self.eps_orig, 2. * self.eps_orig, 1. * self.eps_orig]
            iters = [.3 * self.n_iter_orig, .3 * self.n_iter_orig,
                .4 * self.n_iter_orig]
            iters = [math.ceil(c) for c in iters]
            iters[-1] = self.n_iter_orig - sum(iters[:-1])
            if self.verbose:
                print('using schedule [{}x{}]'.format('+'.join([str(c
                    ) for c in epss]), '+'.join([str(c) for c in iters])))
        
        for target_class in range(2, self.n_target_classes + 2):
            for counter in range(self.n_restarts):
                ind_to_fool = acc.nonzero().squeeze()
                if len(ind_to_fool.shape) == 0:
                    ind_to_fool = ind_to_fool.unsqueeze(0)
                if ind_to_fool.numel() != 0:
                    x_to_fool = x[ind_to_fool].clone()
                    y_to_fool = y[ind_to_fool].clone()
                    
                    if not self.is_tf_model:
                        output = self.model(x_to_fool)
                    else:
                        output = self.model.predict(x_to_fool)
                    self.y_target = output.sort(dim=1)[1][:, -target_class]

                    if not self.use_largereps:
                        res_curr = self.attack_single_run(x_to_fool, y_to_fool)
                    else:
                        res_curr = self.decr_eps_pgd(x_to_fool, y_to_fool, epss, iters)
                    best_curr, acc_curr, loss_curr, adv_curr = res_curr
                    ind_curr = (acc_curr == 0).nonzero().squeeze()

                    acc[ind_to_fool[ind_curr]] = 0
                    adv[ind_to_fool[ind_curr]] = adv_curr[ind_curr].clone()
                    if self.verbose:
                        print('target class {}'.format(target_class),
                            '- restart {} - robust accuracy: {:.2%}'.format(
                            counter, acc.float().mean()),
                            '- cum. time: {:.1f} s'.format(
                            time.time() - startt))

        return adv
</file>

<file path="checks.py">
import torch
import warnings
import math
import sys

from autoattack.other_utils import L2_norm


funcs = {'grad': 0,
    'backward': 0,
    #'enable_grad': 0
    '_make_grads': 0,
    }

checks_doc_path = 'flags_doc.md'


def check_randomized(model, x, y, bs=250, n=5, alpha=1e-4, logger=None):
    acc = []
    corrcl = []
    outputs = []
    with torch.no_grad():
        for _ in range(n):
            output = model(x)
            corrcl_curr = (output.max(1)[1] == y).sum().item()
            corrcl.append(corrcl_curr)
            outputs.append(output / (L2_norm(output, keepdim=True) + 1e-10))
    acc = [c != corrcl_curr for c in corrcl]
    max_diff = 0.
    for c in range(n - 1):
        for e in range(c + 1, n):
            diff = L2_norm(outputs[c] - outputs[e])
            max_diff = max(max_diff, diff.max().item())
            #print(diff.max().item(), max_diff)
    if any(acc) or max_diff > alpha:
        msg = 'it seems to be a randomized defense! Please use version="rand".' + \
            f' See {checks_doc_path} for details.'
        if logger is None:
            warnings.warn(Warning(msg))
        else:
            logger.log(f'Warning: {msg}')


def check_range_output(model, x, alpha=1e-5, logger=None):
    with torch.no_grad():
        output = model(x)
    fl = [output.max() < 1. + alpha, output.min() >  -alpha,
        ((output.sum(-1) - 1.).abs() < alpha).all()]
    if all(fl):
        msg = 'it seems that the output is a probability distribution,' +\
            ' please be sure that the logits are used!' + \
            f' See {checks_doc_path} for details.'
        if logger is None:
            warnings.warn(Warning(msg))
        else:
            logger.log(f'Warning: {msg}')
    return output.shape[-1]


def check_zero_gradients(grad, logger=None):
    z = grad.view(grad.shape[0], -1).abs().sum(-1)
    #print(grad[0, :10])
    if (z == 0).any():
        msg = f'there are {(z == 0).sum()} points with zero gradient!' + \
            ' This might lead to unreliable evaluation with gradient-based attacks.' + \
            f' See {checks_doc_path} for details.'
        if logger is None:
            warnings.warn(Warning(msg))
        else:
            logger.log(f'Warning: {msg}')


def check_square_sr(acc_dict, alpha=.002, logger=None):
    if 'square' in acc_dict.keys() and len(acc_dict) > 2:
        acc = min([v for k, v in acc_dict.items() if k != 'square'])
        if acc_dict['square'] < acc - alpha:
            msg = 'Square Attack has decreased the robust accuracy of' + \
                f' {acc - acc_dict["square"]:.2%}.' + \
                ' This might indicate that the robustness evaluation using' +\
                ' AutoAttack is unreliable. Consider running Square' +\
                ' Attack with more iterations and restarts or an adaptive attack.' + \
                f' See {checks_doc_path} for details.'
            if logger is None:
                warnings.warn(Warning(msg))
            else:
                logger.log(f'Warning: {msg}')


''' from https://stackoverflow.com/questions/26119521/counting-function-calls-python '''
def tracefunc(frame, event, args):
    if event == 'call' and frame.f_code.co_name in funcs.keys():
        funcs[frame.f_code.co_name] += 1

        
def check_dynamic(model, x, is_tf_model=False, logger=None):
    if is_tf_model:
        msg = 'the check for dynamic defenses is not currently supported'
    else:
        msg = None
        sys.settrace(tracefunc)
        model(x)
        sys.settrace(None)
        #for k, v in funcs.items():
        #    print(k, v)
        if any([c > 0 for c in funcs.values()]):
            msg = 'it seems to be a dynamic defense! The evaluation' + \
                ' with AutoAttack might be insufficient.' + \
                f' See {checks_doc_path} for details.'
    if not msg is None:
        if logger is None:
            warnings.warn(Warning(msg))
        else:
            logger.log(f'Warning: {msg}')
    #sys.settrace(None)


def check_n_classes(n_cls, attacks_to_run, apgd_targets, fab_targets,
    logger=None):
    msg = None
    if 'apgd-dlr' in attacks_to_run or 'apgd-t' in attacks_to_run:
        if n_cls <= 2:
            msg = f'with only {n_cls} classes it is not possible to use the DLR loss!'
        elif n_cls == 3:
            msg = f'with only {n_cls} classes it is not possible to use the targeted DLR loss!'
        elif 'apgd-t' in attacks_to_run and \
            apgd_targets + 1 > n_cls:
            msg = f'it seems that more target classes ({apgd_targets})' + \
                f' than possible ({n_cls - 1}) are used in {"apgd-t".upper()}!'
    if 'fab-t' in attacks_to_run and fab_targets + 1 > n_cls:
        if msg is None:
            msg = f'it seems that more target classes ({apgd_targets})' + \
                f' than possible ({n_cls - 1}) are used in FAB-T!'
        else:
            msg += f' Also, it seems that too many target classes ({apgd_targets})' + \
                f' are used in {"fab-t".upper()} ({n_cls - 1} possible)!'
    if not msg is None:
        if logger is None:
            warnings.warn(Warning(msg))
        else:
            logger.log(f'Warning: {msg}')
</file>

<file path="DataManagerPytorch.py">
#DataManagerPytorch 
#Current Version Number = 1.1 (July 15, 2022), Please do not remove this comment.
import torch 
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import math 
from random import shuffle

#Convert an image dataloader (I) to a repeat encoding dataloader (E)
def DataLoaderIToDataLoaderRE(dataLoaderI, length):
    #First convert the image dataloader to tensor form
    xTensor, yTensor = DataLoaderToTensor(dataLoaderI)
    #Create memory for the new tensor with repeat encoding 
    xTensorRepeat = torch.zeros(xTensor.shape + (length,))
    #Go through and fill in the new array, probably a faster way to do this with Pytorch tensors
    for i in range(0, xTensor.shape[0]):
        for j in range(0, length):
            xTensorRepeat[i, :, :, :, j] = xTensor[i]
    #New tensor is filled in, convert back to dataloader
    dataLoaderRE = TensorToDataLoader(xTensorRepeat, yTensor, transforms=None, batchSize =dataLoaderI.batch_size, randomizer = None)
    return dataLoaderRE

#Convert an image dataloader (I) to a repeat encoding dataloader (E)
def DataLoaderREToDataLoaderI(dataLoaderRE):
    #First convert the repeated dataloader to tensor form
    xTensor, yTensor = DataLoaderToTensor(dataLoaderRE)
    #Create memory for the new tensor with repeat encoding 
    xTensorImages = torch.zeros(xTensor.shape[0], xTensor.shape[1], xTensor.shape[2], xTensor.shape[3])
    #Go through and fill in the new array, probably a faster way to do this with Pytorch tensors
    for i in range(0, xTensor.shape[0]):
        xTensorImages[i] = xTensor[i, :, :, :, 0] #Just take the first image from the repeated tensor because they should be the same
    #New tensor is filled in, convert back to dataloader
    dataLoaderI = TensorToDataLoader(xTensorImages, yTensor, transforms=None, batchSize =dataLoaderRE.batch_size, randomizer = None)
    return dataLoaderI

def CheckCudaMem():
    r = torch.cuda.memory_reserved(0) 
    a = torch.cuda.memory_allocated(0)
    f = r-a  # free inside reserved
    print("Unfree Memory=", a)

#Class to help with converting between dataloader and pytorch tensor 
class MyDataSet(torch.utils.data.Dataset):
    def __init__(self, x_tensor, y_tensor, transforms=None):
        self.x = x_tensor
        self.y = y_tensor
        self.transforms = transforms

    def __getitem__(self, index):
        if self.transforms is None: #No transform so return the data directly
            return (self.x[index], self.y[index])
        else: #Transform so apply it to the data before returning 
            return (self.transforms(self.x[index]), self.y[index])

    def __len__(self):
        return len(self.x)

#Validate using a dataloader 
def validateD(valLoader, model, device=None):
    #switch to evaluate mode
    model.eval()
    acc = 0 
    batchTracker = 0
    with torch.no_grad():
        #Go through and process the data in batches 
        for i, (input, target) in enumerate(valLoader):
            sampleSize = input.shape[0] #Get the number of samples used in each batch
            batchTracker = batchTracker + sampleSize
            #print("Processing up to sample=", batchTracker)
            if device == None: #assume cuda
                inputVar = input.cuda()
            else:
                inputVar = input.to(device)
            #compute output
            output = model(inputVar)
            output = output.float()
            #Go through and check how many samples correctly identified
            for j in range(0, sampleSize):
                if output[j].argmax(axis=0) == target[j]:
                    acc = acc +1
    acc = acc / float(len(valLoader.dataset))
    return acc

#Method to validate data using Pytorch tensor inputs and a Pytorch model 
def validateT(xData, yData, model, batchSize=None):
    acc = 0 #validation accuracy 
    numSamples = xData.shape[0]
    model.eval() #change to eval mode
    if batchSize == None: #No batch size so we can feed everything into the GPU
         output = model(xData)
         for i in range(0, numSamples):
             if output[i].argmax(axis=0) == yData[i]:
                 acc = acc+ 1
    else: #There are too many samples so we must process in batch
        numBatches = int(math.ceil(xData.shape[0] / batchSize)) #get the number of batches and type cast to int
        for i in range(0, numBatches): #Go through each batch 
            print(i)
            modelOutputIndex = 0 #reset output index
            startIndex = i*batchSize
            #change the end index depending on whether we are on the last batch or not:
            if i == numBatches-1: #last batch so go to the end
                endIndex = numSamples
            else: #Not the last batch so index normally
                endIndex = (i+1)*batchSize
            output = model(xData[startIndex:endIndex])
            for j in range(startIndex, endIndex): #check how many samples in the batch match the target
                if output[modelOutputIndex].argmax(axis=0) == yData[j]:
                    acc = acc+ 1
                modelOutputIndex = modelOutputIndex + 1 #update the output index regardless
    #Do final averaging and return 
    acc = acc / numSamples
    return acc

#Input a dataloader and model
#Instead of returning a model, output is array with 1.0 dentoting the sample was correctly identified
def validateDA(valLoader, model, device=None):
    numSamples = len(valLoader.dataset)
    accuracyArray = torch.zeros(numSamples) #variable for keep tracking of the correctly identified samples 
    #switch to evaluate mode
    model.eval()
    indexer = 0
    accuracy = 0
    batchTracker = 0
    with torch.no_grad():
        #Go through and process the data in batches 
        for i, (input, target) in enumerate(valLoader):
            sampleSize = input.shape[0] #Get the number of samples used in each batch
            batchTracker = batchTracker + sampleSize
            #print("Processing up to sample=", batchTracker)
            if device == None: #assume CUDA by default
                inputVar = input.cpu() #.cuda()
            else:
                inputVar = input.to(device) #use the prefered device if one is specified
            #compute output
            output = model(inputVar)
            output = output.float()
            #Go through and check how many samples correctly identified
            for j in range(0, sampleSize):
                if output[j].argmax(axis=0) == target[j]:
                    accuracyArray[indexer] = 1.0 #Mark with a 1.0 if sample is correctly identified
                    accuracy = accuracy + 1
                indexer = indexer + 1 #update the indexer regardless 
    accuracy = accuracy/numSamples
    print("Accuracy:", accuracy)
    return accuracyArray

#Replicate TF's predict method behavior 
def predictD(dataLoader, numClasses, model, device=None):
    numSamples = len(dataLoader.dataset)
    yPred = torch.zeros(numSamples, numClasses)
    #switch to evaluate mode
    model.eval()
    indexer = 0
    batchTracker = 0
    with torch.no_grad():
        #Go through and process the data in batches 
        for i, (input, target) in enumerate(dataLoader):
            sampleSize = input.shape[0] #Get the number of samples used in each batch
            batchTracker = batchTracker + sampleSize
            #print("Processing up to sample=", batchTracker)
            if device == None:
                inputVar = input.cuda()
            else:
                inputVar = input.to(device)
            #compute output
            output = model(inputVar)
            output = output.float()
            for j in range(0, sampleSize):
                yPred[indexer] = output[j]
                indexer = indexer + 1 #update the indexer regardless 
    return yPred

#Convert a X and Y tensors into a dataloader
#Does not put any transforms with the data  
def TensorToDataLoader(xData, yData, transforms= None, batchSize=None, randomizer = None):
    if batchSize is None: #If no batch size put all the data through 
        batchSize = xData.shape[0]
    dataset = MyDataSet(xData, yData, transforms)
    if randomizer == None: #No randomizer
        dataLoader = torch.utils.data.DataLoader(dataset=dataset,  batch_size=batchSize, shuffle=False)
    else: #randomizer needed 
        train_sampler = torch.utils.data.RandomSampler(dataset)
        dataLoader = torch.utils.data.DataLoader(dataset=dataset,  batch_size=batchSize, sampler=train_sampler, shuffle=False)
    return dataLoader

#Convert a dataloader into x and y tensors 
def DataLoaderToTensor(dataLoader):
    #First check how many samples in the dataset
    numSamples = len(dataLoader.dataset) 
    sampleShape = GetOutputShape(dataLoader) #Get the output shape from the dataloader
    sampleIndex = 0
    #xData = torch.zeros(numSamples, sampleShape[0], sampleShape[1], sampleShape[2])
    xData = torch.zeros((numSamples,) + sampleShape) #Make it generic shape for non-image datasets
    yData = torch.zeros(numSamples)
    #Go through and process the data in batches 
    for i, (input, target) in enumerate(dataLoader):
        batchSize = input.shape[0] #Get the number of samples used in each batch
        #Save the samples from the batch in a separate tensor 
        for batchIndex in range(0, batchSize):
            xData[sampleIndex] = input[batchIndex]
            yData[sampleIndex] = target[batchIndex]
            sampleIndex = sampleIndex + 1 #increment the sample index 
    return xData, yData 

#Get the output shape from the dataloader
def GetOutputShape(dataLoader):
    for i, (input, target) in enumerate(dataLoader):
        return input[0].shape

#This method randomly creates fake labels for the attack 
#The fake target is guaranteed to not be the same as the original class label 
def GenerateTargetsLabelRandomly(yData, numClasses):
    fTargetLabels=torch.zeros(len(yData))
    for i in range(0, len(yData)):
        targetLabel=random.randint(0,numClasses-1)
        while targetLabel==yData[i]:#Target and true label should not be the same 
            targetLabel=random.randint(0,numClasses-1) #Keep flipping until a different label is achieved 
        fTargetLabels[i]=targetLabel
    return fTargetLabels

#Return the first n correctly classified examples from a model 
#Note examples may not be class balanced 
def GetFirstCorrectlyIdentifiedExamples(device, dataLoader, model, numSamples):
    #First check how many samples in the dataset
    numSamplesTotal = len(dataLoader.dataset) 
    sampleShape = GetOutputShape(dataLoader) #Get the output shape from the dataloader
    sampleIndex = 0
    #xClean = torch.zeros(numSamples, sampleShape[0], sampleShape[1], sampleShape[2])
    xClean = torch.zeros((numSamples,) + sampleShape)
    yClean = torch.zeros(numSamples)
    #switch to evaluate mode
    model.eval()
    acc = 0 
    with torch.no_grad():
        #Go through and process the data in batches 
        for i, (input, target) in enumerate(dataLoader):
            batchSize = input.shape[0] #Get the number of samples used in each batch
            inputVar = input.to(device)
            #compute output
            output = model(inputVar)
            output = output.float()
            #Go through and check how many samples correctly identified
            for j in range(0, batchSize):
                #Add the sample if it is correctly identified and we are not at the limit
                if output[j].argmax(axis=0) == target[j] and sampleIndex<numSamples: 
                    xClean[sampleIndex] = input[j]
                    yClean[sampleIndex] = target[j]
                    sampleIndex = sampleIndex+1
    #Done collecting samples, time to covert to dataloader 
    cleanLoader = TensorToDataLoader(xClean, yClean, transforms=None, batchSize = dataLoader.batch_size, randomizer = None)
    return cleanLoader

def GetCorrectlyIdentifiedSamplesBalanced(model, totalSamplesRequired, dataLoader, numClasses, device=None):
    sampleShape = GetOutputShape(dataLoader)
    xData, yData = DataLoaderToTensor(dataLoader)
    #Basic error checking 
    if totalSamplesRequired % numClasses != 0:
        raise ValueError("The total number of samples in not evenly divisable by the number of classes.")
    #Get the number of samples needed for each class
    numSamplesPerClass = int(totalSamplesRequired/numClasses) 
    correctlyClassifiedSamples = torch.zeros((numClasses, numSamplesPerClass, sampleShape[0], sampleShape[1], sampleShape[2]))
    sanityCounter = torch.zeros((numClasses))
    #yPred = model.predict(xData)
    yPred = predictD(dataLoader, numClasses, model, device)
    for i in range(0, xData.shape[0]): #Go through every sample 
        predictedClass = yPred[i].argmax(axis=0)
        trueClass = yData[i]#.argmax(axis=0) 
        currentSavedCount = int(sanityCounter[int(trueClass)]) #Check how may samples we previously saved from this class
        #If the network predicts the sample correctly and we haven't saved enough samples from this class yet then save it
        if predictedClass == trueClass and currentSavedCount<numSamplesPerClass:
            correctlyClassifiedSamples[int(trueClass), currentSavedCount] = xData[i] #Save the sample 
            sanityCounter[int(trueClass)] = sanityCounter[int(trueClass)] + 1 #Add one to the count of saved samples for this class
    #Now we have gone through the entire network, make sure we have enough samples
    for c in range(0, numClasses):
        if sanityCounter[c] != numSamplesPerClass:
            raise ValueError("The network does not have enough correctly predicted samples for this class.")
    #Assume we have enough samples now, restore in a properly shaped array 
    xCorrect = torch.zeros((totalSamplesRequired, xData.shape[1], xData.shape[2], xData.shape[3]))
    yCorrect = torch.zeros((totalSamplesRequired))
    currentIndex = 0 #indexing for the final array
    for c in range(0, numClasses): #Go through each class
        for j in range(0, numSamplesPerClass): #For each sample in the class store it 
            xCorrect[currentIndex] = correctlyClassifiedSamples[c,j]
            yCorrect[currentIndex] = c
            #yCorrect[currentIndex, c] = 1.0
            currentIndex = currentIndex + 1 
    #return xCorrect, yCorrect
    cleanDataLoader = TensorToDataLoader(xCorrect, yCorrect, transforms = None, batchSize = dataLoader.batch_size, randomizer = None)
    return cleanDataLoader

def GetCorrectlyIdentifiedSamplesBalancedDefense(defense, totalSamplesRequired, dataLoader, numClasses, device):
    sampleShape = GetOutputShape(dataLoader)
    xData, yData = DataLoaderToTensor(dataLoader)
    #Basic error checking 
    if totalSamplesRequired % numClasses != 0:
        raise ValueError("The total number of samples in not evenly divisable by the number of classes.")
    #Get the number of samples needed for each class
    numSamplesPerClass = int(totalSamplesRequired/numClasses) 
    #correctlyClassifiedSamples = torch.zeros((numClasses, numSamplesPerClass, sampleShape[0], sampleShape[1], sampleShape[2]))
    correctlyClassifiedSamples = torch.zeros(((numClasses,) + (numSamplesPerClass,) + sampleShape))
    sanityCounter = torch.zeros((numClasses))
    #yPred = model.predict(xData)
    yPred = defense.predictD(dataLoader, numClasses, device)
    for i in range(0, xData.shape[0]): #Go through every sample 
        predictedClass = yPred[i].argmax(axis=0)
        trueClass = yData[i]#.argmax(axis=0) 
        currentSavedCount = int(sanityCounter[int(trueClass)]) #Check how may samples we previously saved from this class
        #If the network predicts the sample correctly and we haven't saved enough samples from this class yet then save it
        if predictedClass == trueClass and currentSavedCount<numSamplesPerClass:
            correctlyClassifiedSamples[int(trueClass), currentSavedCount] = xData[i] #Save the sample 
            sanityCounter[int(trueClass)] = sanityCounter[int(trueClass)] + 1 #Add one to the count of saved samples for this class
    #Now we have gone through the entire network, make sure we have enough samples
    for c in range(0, numClasses):
        if sanityCounter[c] != numSamplesPerClass:
            raise ValueError("The network does not have enough correctly predicted samples for this class.")
    #Assume we have enough samples now, restore in a properly shaped array 
    #xCorrect = torch.zeros((totalSamplesRequired, xData.shape[1], xData.shape[2], xData.shape[3]))
    xCorrect = torch.zeros(((totalSamplesRequired,) + sampleShape))
    yCorrect = torch.zeros((totalSamplesRequired))
    currentIndex = 0 #indexing for the final array
    for c in range(0, numClasses): #Go through each class
        for j in range(0, numSamplesPerClass): #For each sample in the class store it 
            xCorrect[currentIndex] = correctlyClassifiedSamples[c,j]
            yCorrect[currentIndex] = c
            #yCorrect[currentIndex, c] = 1.0
            currentIndex = currentIndex + 1 
    #return xCorrect, yCorrect
    cleanDataLoader = TensorToDataLoader(xCorrect, yCorrect, transforms = None, batchSize = dataLoader.batch_size, randomizer = None)
    return cleanDataLoader

#Manually shuffle the data loader assuming no transformations
def ManuallyShuffleDataLoader(dataLoader):
    xTest, yTest = DataLoaderToTensor(dataLoader)
    #Shuffle the indicies of the samples 
    indexList = []
    for i in range(0, xTest.shape[0]):
        indexList.append(i)
    shuffle(indexList)
    #Shuffle the samples and put them back in the dataloader 
    xTestShuffle = torch.zeros(xTest.shape)
    yTestShuffle = torch.zeros(yTest.shape)
    for i in range(0, xTest.shape[0]): 
        xTestShuffle[i] = xTest[indexList[i]]
        yTestShuffle[i] = yTest[indexList[i]]
    dataLoaderShuffled = TensorToDataLoader(xTestShuffle, yTestShuffle, transforms = None, batchSize = dataLoader.batch_size, randomizer = None)
    return dataLoaderShuffled
</file>

<file path="other_utils.py">
import os
import collections.abc as container_abcs

import torch

class Logger():
    def __init__(self, log_path):
        self.log_path = log_path
        
    def log(self, str_to_log):
        print(str_to_log)
        if not self.log_path is None:
            with open(self.log_path, 'a') as f:
                f.write(str_to_log + '\n')
                f.flush()
            
def check_imgs(adv, x, norm):
    delta = (adv - x).view(adv.shape[0], -1)
    if norm == 'Linf':
        res = delta.abs().max(dim=1)[0]
    elif norm == 'L2':
        res = (delta ** 2).sum(dim=1).sqrt()
    elif norm == 'L1':
        res = delta.abs().sum(dim=1)

    str_det = 'max {} pert: {:.5f}, nan in imgs: {}, max in imgs: {:.5f}, min in imgs: {:.5f}'.format(
        norm, res.max(), (adv != adv).sum(), adv.max(), adv.min())
    print(str_det)
    
    return str_det

def L1_norm(x, keepdim=False):
    z = x.abs().view(x.shape[0], -1).sum(-1)
    if keepdim:
        z = z.view(-1, *[1]*(len(x.shape) - 1))
    return z

def L2_norm(x, keepdim=False):
    z = (x ** 2).view(x.shape[0], -1).sum(-1).sqrt()
    if keepdim:
        z = z.view(-1, *[1]*(len(x.shape) - 1))
    return z

def L0_norm(x):
    return (x != 0.).view(x.shape[0], -1).sum(-1)

def makedir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def zero_gradients(x):
    if isinstance(x, torch.Tensor):
        if x.grad is not None:
            x.grad.detach_()
            x.grad.zero_()
    elif isinstance(x, container_abcs.Iterable):
        for elem in x:
            zero_gradients(elem)
</file>

</files>
</file>

<file path="attacks/linf_attack/APGD_Linf.py">
#This is the original APGD attack that does not have any gradient masking fixes 
import torch
import utils as DMP
import numpy as np

#This is the operation for DLR Loss function
def DLRLoss(x, y):
    x_sorted, ind_sorted = x.sort(dim=1)
    ind = (ind_sorted[:, -1] == y).float()
    u = torch.arange(x.shape[0])

    return -(x[u, y] - x_sorted[:, -2] * ind - x_sorted[:, -1] * (
        1. - ind)) #/ (x_sorted[:, -1] - x_sorted[:, -3] + 1e-12)

#This operation can all be done in one line but for readability later
#the projection operation is done in multiple steps for l-inf norm
def ProjectionOperation(xAdv, xClean, epsilonMax):
    #First make sure that xAdv does not exceed the acceptable range in the positive direction
    xAdv = torch.min(xAdv, xClean + epsilonMax) 
    #Second make sure that xAdv does not exceed the acceptable range in the negative direction
    xAdv = torch.max(xAdv, xClean - epsilonMax)
    return xAdv

#Function for computing the model gradient
def GetModelGradient(device, model, xK, yK):
    #Define the loss function
    loss = torch.nn.CrossEntropyLoss()
    xK.requires_grad = True
    #Pass the inputs through the model 
    outputs = model(xK.to(device))
    model.zero_grad()
    #Compute the loss 
    cost = loss(outputs, yK)
    cost.backward()
    xKGrad = xK.grad
    return xKGrad

#Function for computing the model gradient for DLR Loss Function
def DLR_GetModelGradient(device, model, xK, yK):
    #Define the loss function
    xK.requires_grad = True
    #Pass the inputs through the model 
    outputs = model(xK.to(device))
    model.zero_grad()
    #Compute the loss 
    cost = DLRLoss(outputs, yK).mean()
    cost.backward()
    xKGrad = xK.grad
    return xKGrad


def ComputePList(pList, startIndex, decrement):
    #p(j+1) = p(j) + max( p(j) - p(j-1) -0.03, 0.06))
    nextP = pList[startIndex] + max(pList[startIndex] - pList[startIndex-1] - decrement, 0.06)
    #Check for base case 
    if nextP>= 1.0:
        return pList
    else:
        #Need to further recur
        pList.append(nextP)
        ComputePList(pList, startIndex+1, decrement)

#Condition two checks if the objective function and step size previously changed
def CheckConditionTwo(f, eta, checkPointIndex, checkPoints):
    currentCheckPoint = checkPoints[checkPointIndex]
    previousCheckPoint = checkPoints[checkPointIndex-1] #Get the previous checkpoint
    if eta[previousCheckPoint] == eta[currentCheckPoint] and f[previousCheckPoint] == f[currentCheckPoint]:
        return True
    else:
        return False

#Condition one checks the summation of objective function
def CheckConditionOne(f, checkPointIndex, checkPoints, targeted):
    sum = 0
    currentCheckPoint = checkPoints[checkPointIndex]
    previousCheckPoint = checkPoints[checkPointIndex-1] #Get the previous checkpoint
    #See how many times the objective function was growing bigger 
    for i in range(previousCheckPoint, currentCheckPoint): #Goes from w_(j-1) to w_(j) - 1
        if f[i+1] > f[i] :
            sum = sum + 1
    ratio = 0.75 * (currentCheckPoint - previousCheckPoint)
    #For untargeted attack we want the objective function to increase
    if targeted == False and sum < ratio: #This is condition 1 from the Autoattack paper
        return True
    elif targeted == True and sum > ratio: #This is my interpretation of how the targeted attack would work (not 100% sure)
        return True
    else:
        return False

def ComputeCheckPoints_New(Niter, decrement, opt=False):
    #First compute the pList based on the decrement amount
    pList = [0, 0.22] #Starting pList based on AutoAttack paper
    ComputePList(pList, 1, decrement)
    #Second compute the checkpoints from the pList
    wList = []
    for i in range(0, len(pList)):
        wList.append(int(np.ceil(pList[i]*Niter)))
    #There may duplicates in the list due to rounding so finally we remove duplicates
    wListFinal = []
    for i in wList:
        if i not in wListFinal:
            wListFinal.append(i)
    #Return the final list
    return wListFinal, {k: v for v, k in enumerate(wListFinal)} if opt else wListFinal

####### Cross_Entropy Loss function

def AutoAttackPytorchMatGPUWrapper(device, dataLoader, model, epsilonMax, etaStart, numSteps, clipMin=0, clipMax=1):
    numSamples = len(dataLoader.dataset) #Get the total number of samples to attack
    xShape = DMP.GetOutputShape(dataLoader) #Get the shape of the input (there may be easier way to do this)
    xAdv = torch.zeros(numSamples, xShape[0], xShape[1], xShape[2])
    yClean = torch.zeros(numSamples)
    batchSize = 0 #just do dummy initalization, will be filled in later
    tracker = 0
    model.eval() #Change model to evaluation mode for the attack 
    #Go through each batch and run the attack
    for xData, yData in dataLoader:
        #Initialize the AutoAttack variables
        batchSize = xData.shape[0] #Get the batch size so we know indexing for saving later
        tracker = tracker + batchSize #Update the tracking variable 
        print(tracker, end = "\r")
        xBest = AutoAttackPytorchMatGPU(device, xData, yData.long(), model, epsilonMax, etaStart, numSteps, clipMin, clipMax)
        xAdv[tracker-batchSize: tracker] = xBest
        yClean[tracker-batchSize: tracker] = yData
    advLoader = DMP.TensorToDataLoader(xAdv, yClean.long(), transforms= None, batchSize= dataLoader.batch_size, randomizer=None) #use the same batch size as the original loader
    return advLoader

def AutoAttackPytorchMatGPU(device, xData, yData, model, epsilonMax, etaStart, numSteps, clipMin=0, clipMax=1): ### only for 1 batch and opt for memory
    #Setup attack variables:
    decrement = 0.03
    wList, wListIndex = ComputeCheckPoints_New(numSteps, decrement, True) #Get the list of checkpoints based on the number of iterations 
    alpha = 0.75 #Weighting factor for momentum 

    # model.eval() #Change model to evaluation mode for the attack 
    batchSize = xData.shape[0] #Get the batch size so we know indexing for saving later
    xShape = xData[0].shape
    
    xData = xData.to(device)
    yK = yData.to(device) #Correct class labels which don't change in the iterations
    eta = torch.zeros(numSteps + 1, batchSize).to(device) #Keep track of the step size for each sample
    eta[0, :] = etaStart #Initalize eta values as the starting eta for each sample in the batch 
    f = torch.zeros(numSteps + 1 , batchSize).to(device) #Keep track of the function value for every sample at every step
    x = torch.zeros(3, batchSize, xShape[0], xShape[1], xShape[2]).to(device)
    x[0] = xData #Initalize the starting adversarial example as the clean example
    lossIndividual = torch.nn.CrossEntropyLoss()
    
    #Do the attack for a number of steps
    for k in range(0, numSteps):
        #First attack step handled slightly differently
        if k == 0:
            xKGrad = GetModelGradient(device, model, x[0], yK) #Get the model gradient 
            x[1] = x[0] + eta[0][:, None, None, None] * torch.sign(xKGrad) #here we use index 1 because the 0th index is the clean sample
            x[1] = torch.clamp(ProjectionOperation(x[1], x[0], epsilonMax), min=clipMin, max=clipMax) #Apply the projection operation and clipping to make sure xAdv does not go out of the adversarial bounds
                
            #Check which adversarial x is better, the clean x or the new adversarial x 
            with torch.no_grad():
                outputsOriginal = model(x[0].to(device)) 
                f[0] = lossIndividual(outputsOriginal, yK).detach() #Store the value in the objective function array
                outputs = model(x[1].to(device)) 
                f[1] = lossIndividual(outputs, yK).detach() #Store the value in the objective function array
                    
            values, indices = torch.max(f[0:2], dim=0)
            xBest = torch.stack([x[indices[i],i] for i in range(batchSize)])
            fBest = values
            #Give a non-zero step size for the next iteration
            eta[1] = eta[0]
                
        #Not the first iteration of the attack
        else:
            xKGrad = GetModelGradient(device, model, x[1], yK) 
            #Compute zk
            z = x[1] + eta[k][:, None, None, None] * torch.sign(xKGrad)
            z = ProjectionOperation(z, xData, epsilonMax)
            #Compute x(k+1) using momentum
            x[2] = x[1] + alpha *(z-x[1]) + (1-alpha)*(x[1]-x[0])
            x[2] =  ProjectionOperation(x[2], xData, epsilonMax)          
            #Apply the clipping operation to make sure xAdv remains in the valid image range
            x[2] = torch.clamp(x[2], min=clipMin, max=clipMax)
        
            #Check which x is better
            with torch.no_grad():
                outputs = model(x[2].to(device))
                f[k + 1] = lossIndividual(outputs, yK).detach()
                print("f[k+1]: ", f[k+1])
                print("fBest: ", fBest)
                
            for b in range(0, batchSize):
                #In the untargeted case we want the cost to increase
                if f[k+1, b] >= fBest[b]: 
                    xBest[b] = x[2, b]
            fBest = torch.maximum(f[k + 1],fBest)
            
            #Now time to do the conditional check to possibly update the step size 
            if k in wListIndex: 
                checkPointIndex = wListIndex[k] #Get the index of the currentCheckpoint
                #Go through each element in the batch 
                for b in range(0, batchSize):
                    conditionOneBoolean = CheckConditionOne(f[:,b], checkPointIndex, wList, False)
                    conditionTwoBoolean = CheckConditionTwo(f[:,b], eta[:,b], checkPointIndex, wList)
                    #If either condition is true halve the step size, else use the step size of the last iteration
                    if conditionOneBoolean == True or conditionTwoBoolean == True:           
                        eta[k + 1, b] = eta[k, b] / 2.0
                    else:
                        eta[k + 1, b] = eta[k, b]
            #If we don't need to check the conditions, just repeat the previous iteration's step size
            else:
                eta[k + 1] = eta[k] 
            
            #Save x[k] to x[k-1], x[k+1] to x[k] for the next k
            x[0],x[1] = x[1],x[2]
        #Memory clean up
        torch.cuda.empty_cache() 
    return xBest


####### DLR Loss function

def DLR_AutoAttackPytorchMatGPUWrapper(device, dataLoader, model, epsilonMax, etaStart, numSteps, clipMin=0, clipMax=1):
    numSamples = len(dataLoader.dataset) #Get the total number of samples to attack
    xShape = DMP.GetOutputShape(dataLoader) #Get the shape of the input (there may be easier way to do this)
    xAdv = torch.zeros(numSamples, xShape[0], xShape[1], xShape[2])
    yClean = torch.zeros(numSamples)
    batchSize = 0 #just do dummy initalization, will be filled in later
    tracker = 0
    model.eval() #Change model to evaluation mode for the attack 
    #Go through each batch and run the attack
    for xData, yData in dataLoader:
        #Initialize the AutoAttack variables
        batchSize = xData.shape[0] #Get the batch size so we know indexing for saving later
        tracker = tracker + batchSize #Update the tracking variable 
        print(tracker, end = "\r")
        xBest = DLR_AutoAttackPytorchMatGPU(device, xData, yData.long(), model, epsilonMax, etaStart, numSteps, clipMin, clipMax)
        xAdv[tracker-batchSize: tracker] = xBest
        yClean[tracker-batchSize: tracker] = yData
    advLoader = DMP.TensorToDataLoader(xAdv, yClean.long(), transforms= None, batchSize= dataLoader.batch_size, randomizer=None) #use the same batch size as the original loader
    return advLoader

def DLR_AutoAttackPytorchMatGPU(device, xData, yData, model, epsilonMax, etaStart, numSteps, clipMin=0, clipMax=1): ### only for 1 batch and opt for memory
    #Setup attack variables:
    decrement = 0.03
    wList, wListIndex = ComputeCheckPoints_New(numSteps, decrement, True) #Get the list of checkpoints based on the number of iterations 
    alpha = 0.75 #Weighting factor for momentum 

    # model.eval() #Change model to evaluation mode for the attack 
    batchSize = xData.shape[0] #Get the batch size so we know indexing for saving later
    xShape = xData[0].shape
    
    xData = xData.to(device)
    yK = yData.to(device) #Correct class labels which don't change in the iterations
    eta = torch.zeros(numSteps + 1, batchSize).to(device) #Keep track of the step size for each sample
    eta[0, :] = etaStart #Initalize eta values as the starting eta for each sample in the batch 
    f = torch.zeros(numSteps + 1 , batchSize).to(device) #Keep track of the function value for every sample at every step
    x = torch.zeros(3, batchSize, xShape[0], xShape[1], xShape[2]).to(device)
    x[0] = xData #Initalize the starting adversarial example as the clean example
    
    #Do the attack for a number of steps
    for k in range(0, numSteps):
        #First attack step handled slightly differently
        if k == 0:
            xKGrad = DLR_GetModelGradient(device, model, x[0], yK) #Get the model gradient 
            x[1] = x[0] + eta[0][:, None, None, None] * torch.sign(xKGrad) #here we use index 1 because the 0th index is the clean sample
            x[1] = torch.clamp(ProjectionOperation(x[1], x[0], epsilonMax), min=clipMin, max=clipMax) #Apply the projection operation and clipping to make sure xAdv does not go out of the adversarial bounds
                
            #Check which adversarial x is better, the clean x or the new adversarial x 
            with torch.no_grad():
                outputsOriginal = model(x[0].to(device)) 
                Individual_cost0 = DLRLoss(outputsOriginal, yK).detach() #Store the value in the objective function array
                f[0] = Individual_cost0.mean()
                
                outputs = model(x[1].to(device)) 
                Individual_cost1 = DLRLoss(outputs, yK).detach() #Store the value in the objective function array
                f[1] = Individual_cost1.mean()
                    
            values, indices = torch.max(f[0:2], dim=0)
            xBest = torch.stack([x[indices[i],i] for i in range(batchSize)])
            fBest = values
            #Give a non-zero step size for the next iteration
            eta[1] = eta[0]
                
        #Not the first iteration of the attack
        else:
            xKGrad = DLR_GetModelGradient(device, model, x[1], yK) 
            #Compute zk
            z = x[1] + eta[k][:, None, None, None] * torch.sign(xKGrad)
            z = ProjectionOperation(z, xData, epsilonMax)
            #Compute x(k+1) using momentum
            x[2] = x[1] + alpha *(z-x[1]) + (1-alpha)*(x[1]-x[0])
            x[2] =  ProjectionOperation(x[2], xData, epsilonMax)          
            #Apply the clipping operation to make sure xAdv remains in the valid image range
            x[2] = torch.clamp(x[2], min=clipMin, max=clipMax)
        
            #Check which x is better
            with torch.no_grad():
                outputs = model(x[2].to(device))
                Individual_costk = DLRLoss(outputs, yK).detach()
                f[k + 1] = Individual_costk.mean()
            for b in range(0, batchSize):
                #In the untargeted case we want the cost to increase
                if f[k+1, b] >= fBest[b]: 
                    xBest[b] = x[2, b]
            fBest = torch.maximum(f[k + 1],fBest)
            
            #Now time to do the conditional check to possibly update the step size 
            if k in wListIndex: 
                checkPointIndex = wListIndex[k] #Get the index of the currentCheckpoint
                #Go through each element in the batch 
                for b in range(0, batchSize):
                    conditionOneBoolean = CheckConditionOne(f[:,b], checkPointIndex, wList, False)
                    conditionTwoBoolean = CheckConditionTwo(f[:,b], eta[:,b], checkPointIndex, wList)
                    #If either condition is true halve the step size, else use the step size of the last iteration
                    if conditionOneBoolean == True or conditionTwoBoolean == True:           
                        eta[k + 1, b] = eta[k, b] / 2.0
                    else:
                        eta[k + 1, b] = eta[k, b]
            #If we don't need to check the conditions, just repeat the previous iteration's step size
            else:
                eta[k + 1] = eta[k] 
            
            #Save x[k] to x[k-1], x[k+1] to x[k] for the next k
            x[0],x[1] = x[1],x[2]
        #Memory clean up
        torch.cuda.empty_cache() 
    return xBest
</file>

<file path="attacks/linf_attack/FGSM.py">
#Attack wrappers class for FGSM and MIM (no extra library implementation) to be used in conjunction with 
#the adaptive black-box attack 
import torch 
import utils as DMP
import torchvision

#Native (no attack library) implementation of the FGSM attack in Pytorch 
def FGSMNativePytorch(device, dataLoader, model, epsilonMax, clipMin, clipMax, targeted):
    model.eval() #Change model to evaluation mode for the attack 
    #Generate variables for storing the adversarial examples 
    numSamples = len(dataLoader.dataset) #Get the total number of samples to attack
    xShape = DMP.GetOutputShape(dataLoader) #Get the shape of the input (there may be easier way to do this)
    xAdv = torch.zeros(numSamples, xShape[0], xShape[1], xShape[2])
    yClean = torch.zeros(numSamples)
    advSampleIndex = 0 
    batchSize = 0 #just do dummy initalization, will be filled in later
    #Go through each sample 
    tracker = 0
    for xData, yData in dataLoader:
        batchSize = xData.shape[0] #Get the batch size so we know indexing for saving later
        tracker = tracker + batchSize
        #print("Processing up to sample=", tracker)
        #Put the data from the batch onto the device 
        xDataTemp = torch.from_numpy(xData.cpu().detach().numpy()).to(device)
        yData = yData.type(torch.LongTensor).to(device)
        # Set requires_grad attribute of tensor. Important for attack. (Pytorch comment, not mine) 
        xDataTemp.requires_grad = True
        # Forward pass the data through the model
        output = model(xDataTemp)
        # Calculate the loss
        loss = torch.nn.CrossEntropyLoss()
        # Zero all existing gradients
        model.zero_grad()
        # Calculate gradients of model in backward pass
        cost = loss(output, yData).to(device)
        cost.backward()
        # Collect datagrad
        #xDataGrad = xDataTemp.grad.data
        ###Here we actual compute the adversarial sample 
        # Collect the element-wise sign of the data gradient
        signDataGrad = xDataTemp.grad.data.sign()
        # Create the perturbed image by adjusting each pixel of the input image
        #print("xData:", xData.is_cuda)
        #print("SignGrad:", signDataGrad.is_cuda)
        if targeted == True:
            perturbedImage = xData - epsilonMax*signDataGrad.cpu().detach() #Go negative of gradient
        else:
            perturbedImage = xData + epsilonMax*signDataGrad.cpu().detach()
        # Adding clipping to maintain the range
        perturbedImage = torch.clamp(perturbedImage, clipMin, clipMax)
        #Save the adversarial images from the batch 
        for j in range(0, batchSize):
            xAdv[advSampleIndex] = perturbedImage[j]
            yClean[advSampleIndex] = yData[j]
            advSampleIndex = advSampleIndex+1 #increment the sample index
        #Not sure if we need this but do some memory clean up 
        del xDataTemp
        del signDataGrad
        torch.cuda.empty_cache()
    #All samples processed, now time to save in a dataloader and return 
    advLoader = DMP.TensorToDataLoader(xAdv, yClean, transforms= None, batchSize= dataLoader.batch_size, randomizer=None) #use the same batch size as the original loader
    return advLoader
</file>

<file path="model_architecture/spiking_resnet_voter.py">
"""
Spiking ResNet V2 for Voter Dataset
Adapted from the ResNet V2 architecture used in the voter classification task.
Supports grayscale images with flexible dimensions (40×50).

Reference:
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Identity Mappings in Deep Residual Networks. arXiv:1603.05027
"""

import torch
import torch.nn as nn
import torch.nn.init as init
from copy import deepcopy
from spikingjelly.activation_based import functional, neuron, layer

__all__ = [
    "SpikingResNetVoter",
    "spiking_resnet20_voter",
    "spiking_resnet56_voter",
    "spiking_resnet164_voter",
    "spiking_resnet1001_voter",
]


def _weights_init(m):
    """Initialize weights using Kaiming normal."""
    if isinstance(m, layer.Linear) or isinstance(m, layer.Conv2d):
        init.kaiming_normal_(m.weight)


class SpikingBasicBlockVoter(nn.Module):
    """
    Spiking Basic Block for ResNet V2 (Pre-activation) for Voter dataset.
    
    This follows the Keras ResNet V2 architecture:
    BN -> SN (instead of ReLU) -> Conv pattern
    """
    expansion = 1

    def __init__(
        self,
        res_block,
        activation,
        batch_normalization,
        in_planes,
        planes,
        stride,
        norm_layer=None,
        spiking_neuron: callable = None,
        **kwargs,
    ):
        super(SpikingBasicBlockVoter, self).__init__()
        
        if norm_layer is None:
            norm_layer = layer.BatchNorm2d
        
        self.res_block = res_block
        self.activation = activation
        self.batch_normalization = batch_normalization

        # ResNet V2 architecture (pre-activation)
        if res_block == 0:
            self.bn1 = norm_layer(in_planes)
            self.conv1 = layer.Conv2d(in_planes, in_planes, kernel_size=1, stride=stride, bias=True)
        else:
            self.bn1 = norm_layer(planes)
            self.conv1 = layer.Conv2d(planes, in_planes, kernel_size=1, stride=stride, bias=True)

        # Spiking neuron for first activation
        self.sn1 = spiking_neuron(**deepcopy(kwargs)) if activation else None

        self.bn2 = norm_layer(in_planes)
        self.conv2 = layer.Conv2d(in_planes, in_planes, kernel_size=3, stride=1, padding=1, bias=True)
        self.sn2 = spiking_neuron(**deepcopy(kwargs))

        self.bn3 = norm_layer(in_planes)
        self.conv3 = layer.Conv2d(in_planes, planes, kernel_size=1, stride=1, bias=True)
        self.sn3 = spiking_neuron(**deepcopy(kwargs))

        # Skip connection
        self.shortcut = nn.Sequential()
        if res_block == 0:
            self.shortcut = nn.Sequential(
                layer.Conv2d(in_planes, planes, kernel_size=1, stride=stride, bias=True)
            )
        
        # Final activation after residual addition
        self.sn_out = spiking_neuron(**deepcopy(kwargs))

    def forward(self, x):
        # Pre-activation pattern: BN -> SN -> Conv
        if self.activation and self.batch_normalization:
            out = self.conv1(self.sn1(self.bn1(x)))
        elif self.activation and not self.batch_normalization:
            out = self.conv1(self.sn1(x))
        elif not self.activation and self.batch_normalization:
            out = self.conv1(self.bn1(x))
        else:
            out = self.conv1(x)
        
        out = self.conv2(self.sn2(self.bn2(out)))
        out = self.conv3(self.sn3(self.bn3(out)))
        out += self.shortcut(x)
        out = self.sn_out(out)
        
        return out


class SpikingResNetVoter(nn.Module):
    """
    Spiking ResNet V2 for Voter Dataset with flexible dimensions and grayscale support.
    
    Args:
        block: Block type (SpikingBasicBlockVoter)
        num_blocks: List of number of blocks per stage [stage1, stage2, stage3]
        imgH: Image height
        imgW: Image width
        num_classes: Number of output classes
        norm_layer: Normalization layer type
        spiking_neuron: Spiking neuron type (e.g., neuron.IFNode)
        **kwargs: Additional arguments for spiking neuron
    """
    
    def __init__(
        self,
        block,
        num_blocks,
        imgH,
        imgW,
        num_classes=2,
        norm_layer=None,
        spiking_neuron: callable = None,
        init_weights=True,
        **kwargs,
    ):
        super(SpikingResNetVoter, self).__init__()
        
        if norm_layer is None:
            norm_layer = layer.BatchNorm2d
        
        self._norm_layer = norm_layer
        self.num_classes = num_classes
        self.in_planes = 16
        
        # Initial convolution (grayscale input)
        self.conv1 = layer.Conv2d(1, 16, kernel_size=3, stride=1, padding=1, bias=True)
        self.bn1 = norm_layer(16)
        self.sn1 = spiking_neuron(**deepcopy(kwargs))
        
        # Three stages (stacks) following ResNet V2 architecture
        # Stage 1
        in_planes = 16
        self.layer1 = self._make_layer(
            block, 0, in_planes, num_blocks[0], 
            spiking_neuron=spiking_neuron, **kwargs
        )
        
        # Stage 2
        in_planes = 64
        self.layer2 = self._make_layer(
            block, 1, in_planes, num_blocks[1],
            spiking_neuron=spiking_neuron, **kwargs
        )
        
        # Stage 3
        in_planes = 128
        self.layer3 = self._make_layer(
            block, 2, in_planes, num_blocks[2],
            spiking_neuron=spiking_neuron, **kwargs
        )
        
        # Final batch norm
        classifier_input_size = in_planes * 2  # 256
        self.bn2 = norm_layer(classifier_input_size)
        
        # Adaptive average pooling to handle flexible input sizes
        self.avgpool = layer.AdaptiveAvgPool2d((1, 1))
        
        # Calculate flatten size
        with torch.no_grad():
            x = torch.zeros(1, 1, imgH, imgW)
            out = self.sn1(self.bn1(self.conv1(x)))
            out = self.layer1(out)
            out = self.layer2(out)
            out = self.layer3(out)
            out = self.bn2(out)
            out = self.avgpool(out)
            flatten_size = out.view(1, -1).shape[1]
        
        # Classifier
        self.fc = layer.Linear(flatten_size, num_classes)
        
        if init_weights:
            self.apply(_weights_init)

    def _make_layer(
        self,
        block,
        stage_num,
        in_planes,
        num_blocks,
        spiking_neuron: callable = None,
        **kwargs,
    ):
        """Build a stage (stack) of residual blocks."""
        norm_layer = self._norm_layer
        layers = []
        
        for res_block in range(num_blocks):
            # Setup following Keras ResNet V2 pattern
            activation = True
            batch_normalization = True
            strides = 1
            
            if stage_num == 0:
                planes = in_planes * 4  # 64
                if res_block == 0:  # First layer and first stage
                    activation = False
                    batch_normalization = False
            else:
                planes = in_planes * 2  # 128 or 256
                if res_block == 0:  # First layer but not first stage
                    strides = 2  # Downsample
            
            layers.append(
                block(
                    res_block,
                    activation,
                    batch_normalization,
                    in_planes,
                    planes,
                    strides,
                    norm_layer=norm_layer,
                    spiking_neuron=spiking_neuron,
                    **kwargs,
                )
            )
        
        return nn.Sequential(*layers)

    def forward(self, x):
        """
        Forward pass supporting both single-step and multi-step modes.
        
        Single-step: input [N, C, H, W] → output [N, num_classes]
        Multi-step: input [T, N, C, H, W] → output [T, N, num_classes]
        """
        out = self.sn1(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.bn2(out)
        out = self.avgpool(out)
        
        # Handle both single-step and multi-step modes
        if len(out.shape) == 4:  # Single-step: [N, C, H, W]
            out = out.view(out.size(0), -1)  # [N, C]
        elif len(out.shape) == 5:  # Multi-step: [T, N, C, H, W]
            out = out.view(out.shape[0], out.shape[1], -1)  # [T, N, C]
        
        out = self.fc(out)
        return out


# --------------- Factory Functions ---------------

def _spiking_resnet_voter(
    num_blocks,
    imgH,
    imgW,
    num_classes,
    spiking_neuron: callable = None,
    **kwargs,
):
    """Internal factory function."""
    model = SpikingResNetVoter(
        block=SpikingBasicBlockVoter,
        num_blocks=num_blocks,
        imgH=imgH,
        imgW=imgW,
        num_classes=num_classes,
        spiking_neuron=spiking_neuron,
        **kwargs,
    )
    return model


def spiking_resnet20_voter(imgH, imgW, num_classes=2, spiking_neuron: callable = None, **kwargs):
    """Spiking ResNet-20 V2 for Voter dataset."""
    return _spiking_resnet_voter([2, 2, 2], imgH, imgW, num_classes, spiking_neuron, **kwargs)


def spiking_resnet56_voter(imgH, imgW, num_classes=2, spiking_neuron: callable = None, **kwargs):
    """Spiking ResNet-56 V2 for Voter dataset."""
    return _spiking_resnet_voter([6, 6, 6], imgH, imgW, num_classes, spiking_neuron, **kwargs)


def spiking_resnet164_voter(imgH, imgW, num_classes=2, spiking_neuron: callable = None, **kwargs):
    """Spiking ResNet-164 V2 for Voter dataset."""
    return _spiking_resnet_voter([18, 18, 18], imgH, imgW, num_classes, spiking_neuron, **kwargs)


def spiking_resnet1001_voter(imgH, imgW, num_classes=2, spiking_neuron: callable = None, **kwargs):
    """Spiking ResNet-1001 V2 for Voter dataset."""
    return _spiking_resnet_voter([111, 111, 111], imgH, imgW, num_classes, spiking_neuron, **kwargs)
</file>

<file path="model_architecture/spiking_vgg_voter.py">
"""
Spiking VGG for Voter Dataset
Wrapper around generic SpikingVGG with voter-specific defaults.
Supports grayscale images with flexible dimensions (40×50).
"""

import torch
import torch.nn as nn
from copy import deepcopy
from spikingjelly.activation_based import functional, neuron, layer

__all__ = [
    "SpikingVGGVoter",
    "spiking_vgg11_voter",
    "spiking_vgg11_bn_voter",
    "spiking_vgg13_voter",
    "spiking_vgg13_bn_voter",
    "spiking_vgg16_voter",
    "spiking_vgg16_bn_voter",
    "spiking_vgg19_voter",
    "spiking_vgg19_bn_voter",
]


cfg = {
    'VGG11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'VGG19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}


class SpikingVGGVoter(nn.Module):
    """
    Spiking VGG for Voter Dataset with flexible dimensions and grayscale support.
    
    Args:
        vgg_name: 'VGG11', 'VGG13', 'VGG16', or 'VGG19'
        imgH: Image height
        imgW: Image width
        num_classes: Number of output classes
        batch_norm: Whether to use batch normalization
        spiking_neuron: Spiking neuron type (e.g., neuron.IFNode)
        **kwargs: Additional arguments for spiking neuron
    """
    
    def __init__(
        self,
        vgg_name,
        imgH,
        imgW,
        num_classes,
        batch_norm=False,
        norm_layer=None,
        spiking_neuron: callable = None,
        init_weights=True,
        **kwargs,
    ):
        super(SpikingVGGVoter, self).__init__()
        
        if norm_layer is None:
            norm_layer = layer.BatchNorm2d
        
        # Build feature layers
        self.features = self._make_layers(
            cfg[vgg_name],
            batch_norm=batch_norm,
            norm_layer=norm_layer,
            spiking_neuron=spiking_neuron,
            **kwargs,
        )
        
        # Average pool - Adaptive to ensure [N, C, 1, 1] output
        self.avgpool = layer.AdaptiveAvgPool2d((1, 1))
        
        # Calculate flatten size AFTER avgpool
        with torch.no_grad():
            x = torch.zeros(1, 1, imgH, imgW)  # Grayscale input
            out = self.features(x)
            out = self.avgpool(out)  # Apply avgpool to get correct shape
            flatten_size = out.view(1, -1).shape[1]  # Should be num_channels (512)
        
        # Classifier
        self.classifier = layer.Linear(flatten_size, num_classes)
        
        if init_weights:
            self._initialize_weights()

    def forward(self, x):
        """
        Forward pass supporting both single-step and multi-step modes.
        
        Single-step: input [N, C, H, W] → output [N, num_classes]
        Multi-step: input [T, N, C, H, W] → output [T, N, num_classes]
        """
        x = self.features(x)
        x = self.avgpool(x)
        
        # Handle both single-step and multi-step modes
        if len(x.shape) == 4:  # Single-step: [N, C, H, W]
            x = x.view(x.size(0), -1)  # [N, C]
        elif len(x.shape) == 5:  # Multi-step: [T, N, C, H, W]
            x = x.view(x.shape[0], x.shape[1], -1)  # [T, N, C]
        
        x = self.classifier(x)
        return x

    def _initialize_weights(self):
        """Initialize weights using Kaiming normal for Conv and normal for Linear."""
        for m in self.modules():
            if isinstance(m, layer.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, layer.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, layer.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    @staticmethod
    def _make_layers(
        cfg_list,
        batch_norm=False,
        norm_layer=None,
        spiking_neuron: callable = None,
        **kwargs,
    ):
        """Build feature extraction layers."""
        if norm_layer is None:
            norm_layer = layer.BatchNorm2d
        
        layers = []
        in_channels = 1  # Grayscale
        
        for v in cfg_list:
            if v == 'M':
                layers += [layer.MaxPool2d(kernel_size=2, stride=2)]
            else:
                conv2d = layer.Conv2d(in_channels, v, kernel_size=3, padding=1)
                if batch_norm:
                    layers += [
                        conv2d,
                        norm_layer(v),
                        spiking_neuron(**deepcopy(kwargs))
                    ]
                else:
                    layers += [
                        conv2d,
                        spiking_neuron(**deepcopy(kwargs))
                    ]
                in_channels = v
        
        return nn.Sequential(*layers)


# --------------- Factory Functions ---------------

def _spiking_vgg_voter(
    vgg_name,
    imgH,
    imgW,
    num_classes,
    batch_norm=False,
    spiking_neuron: callable = None,
    **kwargs,
):
    """Internal factory function."""
    model = SpikingVGGVoter(
        vgg_name=vgg_name,
        imgH=imgH,
        imgW=imgW,
        num_classes=num_classes,
        batch_norm=batch_norm,
        spiking_neuron=spiking_neuron,
        **kwargs,
    )
    return model


def spiking_vgg11_voter(imgH, imgW, num_classes=2, spiking_neuron: callable = None, **kwargs):
    """Spiking VGG11 without batch normalization for Voter dataset."""
    return _spiking_vgg_voter("VGG11", imgH, imgW, num_classes, False, spiking_neuron, **kwargs)


def spiking_vgg11_bn_voter(imgH, imgW, num_classes=2, spiking_neuron: callable = None, **kwargs):
    """Spiking VGG11 with batch normalization for Voter dataset."""
    return _spiking_vgg_voter("VGG11", imgH, imgW, num_classes, True, spiking_neuron, **kwargs)


def spiking_vgg13_voter(imgH, imgW, num_classes=2, spiking_neuron: callable = None, **kwargs):
    """Spiking VGG13 without batch normalization for Voter dataset."""
    return _spiking_vgg_voter("VGG13", imgH, imgW, num_classes, False, spiking_neuron, **kwargs)


def spiking_vgg13_bn_voter(imgH, imgW, num_classes=2, spiking_neuron: callable = None, **kwargs):
    """Spiking VGG13 with batch normalization for Voter dataset."""
    return _spiking_vgg_voter("VGG13", imgH, imgW, num_classes, True, spiking_neuron, **kwargs)


def spiking_vgg16_voter(imgH, imgW, num_classes=2, spiking_neuron: callable = None, **kwargs):
    """Spiking VGG16 without batch normalization for Voter dataset."""
    return _spiking_vgg_voter("VGG16", imgH, imgW, num_classes, False, spiking_neuron, **kwargs)


def spiking_vgg16_bn_voter(imgH, imgW, num_classes=2, spiking_neuron: callable = None, **kwargs):
    """Spiking VGG16 with batch normalization for Voter dataset."""
    return _spiking_vgg_voter("VGG16", imgH, imgW, num_classes, True, spiking_neuron, **kwargs)


def spiking_vgg19_voter(imgH, imgW, num_classes=2, spiking_neuron: callable = None, **kwargs):
    """Spiking VGG19 without batch normalization for Voter dataset."""
    return _spiking_vgg_voter("VGG19", imgH, imgW, num_classes, False, spiking_neuron, **kwargs)


def spiking_vgg19_bn_voter(imgH, imgW, num_classes=2, spiking_neuron: callable = None, **kwargs):
    """Spiking VGG19 with batch normalization for Voter dataset."""
    return _spiking_vgg_voter("VGG19", imgH, imgW, num_classes, True, spiking_neuron, **kwargs)
</file>

<file path="model_architecture/UNet.py">
import torch
import torch.nn as nn
import torch.nn.functional as F

class UNet(nn.Module):
    def __init__(self):
        super(UNet, self).__init__()

        # --- Encoder ---
        # Block 1: 40x50 -> 20x25
        self.enc1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True)
        )
        self.pool1 = nn.MaxPool2d(2)

        # Block 2: 20x25 -> 10x12
        self.enc2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        self.pool2 = nn.MaxPool2d(2)

        # Block 3: 10x12 -> 5x6
        self.enc3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )
        self.pool3 = nn.MaxPool2d(2)

        # --- Bottleneck ---
        self.bottleneck = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )

        # --- Decoder ---
        # Up 1: 5x6 -> 10x12
        self.up1 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.dec1 = nn.Sequential(
            nn.Conv2d(256, 128, kernel_size=3, padding=1), # 256 input because of concat
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )

        # Up 2: 10x12 -> 20x25
        self.up2 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec2 = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )

        # Up 3: 20x25 -> 40x50
        self.up3 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.dec3 = nn.Sequential(
            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            # Final output layer
            nn.Conv2d(32, 1, kernel_size=1) 
        )

    def forward(self, x):
        # Encoder
        e1 = self.enc1(x)
        p1 = self.pool1(e1)

        e2 = self.enc2(p1)
        p2 = self.pool2(e2)

        e3 = self.enc3(p2)
        p3 = self.pool3(e3)

        # Bottleneck
        b = self.bottleneck(p3)

        # Decoder with Force Alignment
        # We resize 'd1' to match 'e3' exactly because 25/2 is odd and causes shape mismatch
        d1 = self.up1(b)
        if d1.shape != e3.shape: d1 = F.interpolate(d1, size=e3.shape[2:])
        d1 = torch.cat((d1, e3), dim=1) # Skip Connection
        d1 = self.dec1(d1)

        d2 = self.up2(d1)
        if d2.shape != e2.shape: d2 = F.interpolate(d2, size=e2.shape[2:])
        d2 = torch.cat((d2, e2), dim=1) # Skip Connection
        d2 = self.dec2(d2)

        d3 = self.up3(d2)
        if d3.shape != e1.shape: d3 = F.interpolate(d3, size=e1.shape[2:])
        d3 = torch.cat((d3, e1), dim=1) # Skip Connection
        #out = self.dec3(d3)     #orignal has not it. 
        delta = self.dec3(d3)          # raw output, Predict the correction (can be positive or negative) #original has this
        out = torch.clamp(x + delta, 0, 1)  #original has this
        
        return out     #original has this
        #return torch.sigmoid(out)
</file>

<file path="AttackFactory.py">
import torch
import utils

# Import all attacks
from attacks.linf_attack.APGD_Linf import DLR_AutoAttackPytorchMatGPUWrapper as APGD_Linf_Attack
from attacks.l1_attack.autoattack import AutoAttackL1 as AutoAttack
from attacks.l2_attack import autopgd_base
from attacks.l0_attack.L0_PGD import L0_PGD_AttackWrapper
from attacks.l0_attack.L0_Linf_PGD import L0_Linf_PGD_AttackWrapper
from attacks.l0_attack.L0_Sigma_PGD import L0_Sigma_PGD_AttackWrapper


# =========================================================
# Attack 1: APGD-Linf
# =========================================================
def APGD_Linf(device, model, correctLoader, eps):
    """Run APGD-Linf attack and return adversarial loader."""
    clipMin, clipMax = 0.0, 1.0
    numSteps = 500
    etaStart = 2.0 * eps

    print(f"\n--- Running APGD-Linf (DLR) | eps={eps:.6f} ---")
    
    advLoader = APGD_Linf_Attack(
        device, correctLoader, model, eps, etaStart, numSteps, clipMin, clipMax
    )
    
    return advLoader


# =========================================================
# Attack 2: APGD-L1
# =========================================================
def APGD_L1(device, model, correctLoader, eps):
    """Run APGD-L1 attack and return adversarial loader."""
    
    # Convert dataloader to tensors
    x_list, y_list = [], []
    for x, y in correctLoader:
        x_list.append(x)
        y_list.append(y)
    x_data = torch.cat(x_list, dim=0).float()
    y_data = torch.cat(y_list, dim=0).long()

    print(f"\n--- Running APGD-L1 (DLR) | eps={eps} ---")
    
    adv = AutoAttack(
        model,
        eps=eps,
        version="custom",
        attacks_to_run=["apgd-dlr"],
        device=device
    )

    adv.apgd.n_iter = 500
    adv.apgd.n_restarts = 1
    adv.apgd.use_rs = False
    adv.seed = 0

    x_adv = adv.run_standard_evaluation(x_data.to(device), y_data.to(device), bs=64)

    advLoader = utils.TensorToDataLoader(x_adv.detach().cpu(), y_data.cpu(), batchSize=64)
    
    return advLoader


# =========================================================
# Attack 3: APGD-L2
# =========================================================
def APGD_L2(device, model, correctLoader, eps_max):
    """Run APGD-L2 attack and return adversarial loader."""
    num_steps = 200
    loss_name = "dlr"

    print(f"\n--- Running APGD-L2 (DLR) | eps_max={eps_max} ---")

    attackObject = autopgd_base.APGDAttack(
        predict=model,
        n_iter=num_steps,
        norm="L2",
        n_restarts=0,
        eps=eps_max,
        seed=0,
        loss=loss_name,
        eot_iter=1,
        rho=0.75,
        topk=None,
        verbose=False,
        device=device,
        use_largereps=False,
        is_tf_model=False,
        logger=None
    )

    # Initialize hyperparameters
    dummy_x = torch.zeros((1, 1, 40, 50), device=device)
    attackObject.init_hyperparam(dummy_x)

    advLoader = attackObject.APGDCroceAttackWrapper(device, correctLoader)
    
    return advLoader

# Attack 4: L0-PGD
def L0_PGD(device, model, correctLoader, sparsity):
    """Run L0-PGD attack and return adversarial loader."""
    n_restarts = 10
    num_steps = 30
    step_size = 20
    random_start = False

    print(f"\n--- Running L0-PGD | sparsity={sparsity} ---")

    advLoader = L0_PGD_AttackWrapper(
        model=model,
        device=device,
        dataLoader=correctLoader,
        n_restarts=n_restarts,
        num_steps=num_steps,
        step_size=step_size,
        sparsity=sparsity,
        random_start=random_start
    )
    
    return advLoader


# --------- Attack 5: L0 + Linf PGD ------------
def L0_Linf_PGD(device, model, correctLoader, eps):
    """Run L0+Linf PGD attack and return adversarial loader."""
    k = 20
    n_restarts = 10
    step_size = 20
    num_steps = 30
    random_start = False

    print(f"\n--- Running L0+Linf PGD | eps={eps:.6f} ---")

    advLoader = L0_Linf_PGD_AttackWrapper(
        model=model,
        device=device,
        dataLoader=correctLoader,
        n_restarts=n_restarts,
        num_steps=num_steps,
        step_size=step_size,
        sparsity=k,
        epsilon=eps,
        random_start=random_start
    )
    
    return advLoader


# ------ Attack 6: L0 + Sigma-map PGD
def L0_Sigma_PGD(device, model, correctLoader, sparsity):
    """Run L0+Sigma PGD attack and return adversarial loader."""
    n_restarts = 10
    num_steps = 75
    step_size = 15
    kappa = 10
    random_start = False

    print(f"\n--- Running L0+Sigma PGD | sparsity={sparsity} ---")

    advLoader = L0_Sigma_PGD_AttackWrapper(
        model=model,
        device=device,
        dataLoader=correctLoader,
        n_restarts=n_restarts,
        num_steps=num_steps,
        step_size=step_size,
        sparsity=sparsity,
        kappa=kappa,
        random_start=random_start
    )
    
    return advLoader
</file>

<file path="AttackRunner.py">
# AttackRunner.py
# Runs all attacks with multiple parameters, evaluates, saves samples, and writes results

import torch
import os
import utils
import AttackFactory

# Import configurations from config.py
from config import (
    APGD_LINF_PARAMS,
    APGD_L1_PARAMS,
    APGD_L2_PARAMS,
    L0_PGD_PARAMS,
    L0_LINF_PARAMS,
    L0_SIGMA_PARAMS,
    SAVE_CONFIG,
)


# =========================================================
# UTILITY FUNCTIONS
# =========================================================

def evaluate_attack(adv_loader, synthetic_model, oracle, device, num_classes):
    """Evaluate adversarial samples on both models."""
    # Synthetic model
    syn_acc = utils.validateD(adv_loader, synthetic_model, device)
    print(f"[SYNTHETIC MODEL] Adversarial Accuracy: {syn_acc:.4f}")
    utils.calculateClasswiseAccuracy(adv_loader, synthetic_model, device, num_classes)
    
    # Oracle model
    oracle_acc = utils.validateD(adv_loader, oracle, device)
    print(f"[ORACLE MODEL] Adversarial Accuracy: {oracle_acc:.4f}")
    utils.calculateClasswiseAccuracy(adv_loader, oracle, device, num_classes)
    
    return syn_acc, oracle_acc


def save_adv_samples(adv_loader, attack_name, param_name, param_value, n_save=None):
    """Save adversarial samples to organized folder structure."""
    if n_save is None:
        n_save = SAVE_CONFIG["n_save"]
    
    base_dir = os.path.join(SAVE_CONFIG["base_dir"], attack_name)
    os.makedirs(base_dir, exist_ok=True)
    
    # Create filename
    if isinstance(param_value, float) and param_value <= 1:
        filename = f"{param_name}_{int(param_value * 255)}by255.pt"
    else:
        filename = f"{param_name}_{param_value}.pt"
    
    save_path = os.path.join(base_dir, filename)
    
    imgs, lbls = [], []
    count = 0
    for x, y in adv_loader:
        imgs.append(x.cpu())
        lbls.append(y.cpu())
        count += x.size(0)
        if count >= n_save:
            break
    
    final_imgs = torch.cat(imgs)[:n_save]
    final_lbls = torch.cat(lbls)[:n_save]
    torch.save({"images": final_imgs, "labels": final_lbls}, save_path)
    print(f"Saved {final_imgs.shape[0]} samples to {save_path}")
    
    return save_path


def write_attack_results_to_file(results_file, attack_name, param_name, results_summary):
    """Write results for one attack type to file."""
    results_file.write(f"\n\n{'='*90}\n")
    results_file.write(f"{attack_name} RESULTS\n")
    results_file.write(f"{'='*90}\n")
    results_file.write(f"{param_name:<15} {'Value':<15} {'Syn Acc':<12} {'Oracle Acc':<12}\n")
    results_file.write(f"{'-'*90}\n")
    
    for name, results in results_summary.items():
        val = results['param_value']
        val_str = f"{val:.6f}" if isinstance(val, float) and val < 10 else f"{val}"
        
        results_file.write(
            f"{name:<15} {val_str:<15} {results['syn_acc']:<12.4f} {results['oracle_acc']:<12.4f}\n"
        )
    
    results_file.write(f"{'-'*90}\n")
    results_file.flush()


# =========================================================
# MAIN ATTACK RUNNER
# =========================================================

def run_all_attacks(device, oracle, synthetic_model, correct_loader, num_classes, results_file):
    """Run all attacks with all parameter values."""
    
    all_results = {}
    
    # =====================================================
    # ATTACK 1: APGD-Linf
    # =====================================================
    print("\n" + "#"*70)
    print("# RUNNING: APGD-Linf")
    print("#"*70)
    
    apgd_linf_results = {}
    for param_name, eps in APGD_LINF_PARAMS.items():
        print(f"\n{'='*60}")
        print(f"APGD-Linf | eps = {eps}")
        print(f"{'='*60}")
        
        # Run attack
        adv_loader = AttackFactory.APGD_Linf(device, synthetic_model, correct_loader, eps)
        
        # Evaluate
        syn_acc, oracle_acc = evaluate_attack(adv_loader, synthetic_model, oracle, device, num_classes)
        
        # Save samples
        save_adv_samples(adv_loader, "APGD_Linf", "eps", eps)
        
        # Store results
        apgd_linf_results[param_name] = {
            "param_value": eps,
            "syn_acc": syn_acc,
            "oracle_acc": oracle_acc,
        }
    
    write_attack_results_to_file(results_file, "APGD-Linf", "Epsilon", apgd_linf_results)
    all_results["apgd_linf"] = apgd_linf_results
    
    # =====================================================
    # ATTACK 2: APGD-L1
    # =====================================================
    print("\n" + "#"*70)
    print("# RUNNING: APGD-L1")
    print("#"*70)
    
    apgd_l1_results = {}
    for param_name, eps in APGD_L1_PARAMS.items():
        print(f"\n{'='*60}")
        print(f"APGD-L1 | eps = {eps}")
        print(f"{'='*60}")
        
        # Run attack
        adv_loader = AttackFactory.APGD_L1(device, synthetic_model, correct_loader, eps)
        
        # Evaluate
        syn_acc, oracle_acc = evaluate_attack(adv_loader, synthetic_model, oracle, device, num_classes)
        
        # Save samples
        save_adv_samples(adv_loader, "APGD_L1", "eps", eps)
        
        # Store results
        apgd_l1_results[param_name] = {
            "param_value": eps,
            "syn_acc": syn_acc,
            "oracle_acc": oracle_acc,
        }
    
    write_attack_results_to_file(results_file, "APGD-L1", "L1 Epsilon", apgd_l1_results)
    all_results["apgd_l1"] = apgd_l1_results
    
    # =====================================================
    # ATTACK 3: APGD-L2
    # =====================================================
    print("\n" + "#"*70)
    print("# RUNNING: APGD-L2")
    print("#"*70)
    
    apgd_l2_results = {}
    for param_name, eps_max in APGD_L2_PARAMS.items():
        print(f"\n{'='*60}")
        print(f"APGD-L2 | eps_max = {eps_max}")
        print(f"{'='*60}")
        
        # Run attack
        adv_loader = AttackFactory.APGD_L2(device, synthetic_model, correct_loader, eps_max)
        
        # Evaluate
        syn_acc, oracle_acc = evaluate_attack(adv_loader, synthetic_model, oracle, device, num_classes)
        
        # Save samples
        save_adv_samples(adv_loader, "APGD_L2", "eps", eps_max)
        
        # Store results
        apgd_l2_results[param_name] = {
            "param_value": eps_max,
            "syn_acc": syn_acc,
            "oracle_acc": oracle_acc,
        }
    
    write_attack_results_to_file(results_file, "APGD-L2", "L2 Epsilon", apgd_l2_results)
    all_results["apgd_l2"] = apgd_l2_results
    
    # =====================================================
    # ATTACK 4: L0-PGD
    # =====================================================
    print("\n" + "#"*70)
    print("# RUNNING: L0-PGD")
    print("#"*70)
    
    l0_pgd_results = {}
    for param_name, sparsity in L0_PGD_PARAMS.items():
        print(f"\n{'='*60}")
        print(f"L0-PGD | sparsity = {sparsity}")
        print(f"{'='*60}")
        
        # Run attack
        adv_loader = AttackFactory.L0_PGD(device, synthetic_model, correct_loader, sparsity)
        
        # Evaluate
        syn_acc, oracle_acc = evaluate_attack(adv_loader, synthetic_model, oracle, device, num_classes)
        
        # Save samples
        save_adv_samples(adv_loader, "L0_PGD", "k", sparsity)
        
        # Store results
        l0_pgd_results[param_name] = {
            "param_value": sparsity,
            "syn_acc": syn_acc,
            "oracle_acc": oracle_acc,
        }
    
    write_attack_results_to_file(results_file, "L0-PGD", "Sparsity(k)", l0_pgd_results)
    all_results["l0_pgd"] = l0_pgd_results
    
    # =====================================================
    # ATTACK 5: L0+Linf PGD
    # =====================================================
    print("\n" + "#"*70)
    print("# RUNNING: L0+Linf PGD")
    print("#"*70)
    
    l0_linf_results = {}
    for param_name, eps in L0_LINF_PARAMS.items():
        print(f"\n{'='*60}")
        print(f"L0+Linf PGD | eps = {eps}")
        print(f"{'='*60}")
        
        # Run attack
        adv_loader = AttackFactory.L0_Linf_PGD(device, synthetic_model, correct_loader, eps)
        
        # Evaluate
        syn_acc, oracle_acc = evaluate_attack(adv_loader, synthetic_model, oracle, device, num_classes)
        
        # Save samples
        save_adv_samples(adv_loader, "L0_Linf_PGD", "eps", eps)
        
        # Store results
        l0_linf_results[param_name] = {
            "param_value": eps,
            "syn_acc": syn_acc,
            "oracle_acc": oracle_acc,
        }
    
    write_attack_results_to_file(results_file, "L0+Linf PGD", "Epsilon", l0_linf_results)
    all_results["l0_linf"] = l0_linf_results
    
    # =====================================================
    # ATTACK 6: L0+Sigma PGD
    # =====================================================
    print("\n" + "#"*70)
    print("# RUNNING: L0+Sigma PGD")
    print("#"*70)
    
    l0_sigma_results = {}
    for param_name, sparsity in L0_SIGMA_PARAMS.items():
        print(f"\n{'='*60}")
        print(f"L0+Sigma PGD | sparsity = {sparsity}")
        print(f"{'='*60}")
        
        # Run attack
        adv_loader = AttackFactory.L0_Sigma_PGD(device, synthetic_model, correct_loader, sparsity)
        
        # Evaluate
        syn_acc, oracle_acc = evaluate_attack(adv_loader, synthetic_model, oracle, device, num_classes)
        
        # Save samples
        save_adv_samples(adv_loader, "L0_Sigma_PGD", "k", sparsity)
        
        # Store results
        l0_sigma_results[param_name] = {
            "param_value": sparsity,
            "syn_acc": syn_acc,
            "oracle_acc": oracle_acc,
        }
    
    write_attack_results_to_file(results_file, "L0+Sigma PGD", "Sparsity(k)", l0_sigma_results)
    all_results["l0_sigma"] = l0_sigma_results
    
    return all_results
</file>

<file path="config.py">
# config.py
# Centralized configuration for all attack parameters

"""
Attack Parameter Configurations
===============================
Contains all parameter values for each attack type.
"""

# =========================================================
# APGD-Linf: Epsilon values (L∞ norm budget)
# =========================================================
APGD_LINF_PARAMS = {
    "eps_255_255": 255/255,
    "eps_64_255": 64/255,
    "eps_32_255": 32/255,
    "eps_16_255": 16/255,
    "eps_8_255": 8/255,
    "eps_4_255": 4/255,
}

# =========================================================
# APGD-L1: L1 distance values
# =========================================================
APGD_L1_PARAMS = {
    "eps_2000": 2000,
    "eps_200": 200,
    "eps_20": 20,
    "eps_8": 8,
    "eps_4": 4,
    "eps_2": 2,
}

# =========================================================
# APGD-L2: L2 distance values
# =========================================================
APGD_L2_PARAMS = {
    "eps_45": 45,
    "eps_15": 15,
    "eps_5": 5,
    "eps_3": 3,
    "eps_2": 2,
    "eps_1": 1,
}

# =========================================================
# L0-PGD: Sparsity values (number of pixels to modify)
# =========================================================
L0_PGD_PARAMS = {
    "k_2000": 2000,
    "k_500": 500,
    "k_200": 200,
    "k_20": 20,
    "k_10": 10,
    "k_1": 1,
}

# =========================================================
# L0+Linf PGD: Epsilon values
# =========================================================
L0_LINF_PARAMS = {
    "eps_255_255": 255/255,
    "eps_64_255": 64/255,
    "eps_32_255": 32/255,
    "eps_16_255": 16/255,
    "eps_8_255": 8/255,
    "eps_4_255": 4/255,
}

# =========================================================
# L0+Sigma PGD: Sparsity values
# =========================================================
L0_SIGMA_PARAMS = {
    "k_2000": 2000,
    "k_500": 500,
    "k_200": 200,
    "k_20": 20,
    "k_10": 10,
    "k_1": 1,
}

# =========================================================
# SAMPLE SAVING CONFIGURATION
# =========================================================
SAVE_CONFIG = {
    "n_save": 500,
    "base_dir": "adv_samples",
}
</file>

<file path="constants.py">
"""
constants.py

Configuration file containing paths and settings for model experiments.

This file defines:
    - CHECKPOINTS: Paths to pre-trained model checkpoint files
    - VAL_DATASETS: Paths to validation dataset files
    - SCANNED_DATASETS: Paths to scanned bubble datasets (for UNet experiments)
    - EXPERIMENTS: Model-dataset configurations for running experiments
    - EXPERIMENTS_UNET_*: UNet + Model configurations using scanned data
"""

# -------------------------- Checkpoints for all models -----------------------------------------
CHECKPOINTS = {
    "resnet20_combined": "./checkpoint/ModelResNet20-VotingCombined-v2-Grayscale-Run1.th",
    "vgg16_combined": "./checkpoint/ModelVgg16-C2.th",
    "cait_combined": "./checkpoint/ModelCaiT-trCombined-v2-valCombined-v2-Grayscale-Run1.th",
    "svm_combined": [
        "./checkpoint/sklearn_SVM_Combined_v2_Grayscale_Run1/base_pytorch_svm_combined_v2.pth",
        "./checkpoint/sklearn_SVM_Combined_v2_Grayscale_Run1/multi_output_svm_combined_v2.pth",
    ],
    "snn_vgg16_combined": "./checkpoint/spiking_vgg16_bn_voter.pth",
    "snn_resnet20_combined": "./checkpoint/spiking_resnet20_voter.pth",
    "expv2_resnet20": "./checkpoint/Explainable_ResNet20.pth",
    "expv2_vgg16": "./checkpoint/Explainable_VGG16.pth",
    "mambavision_combined": "./checkpoint/mamba_model.pth",
    "unet": "./checkpoint/UNet.th",
}

# ------------------------ UNet Checkpoint ----------------------------------
UNET_CHECKPOINT = CHECKPOINTS["unet"]

# ------------------------ Training and Validation datasets ---------------------------------------
VAL_DATASETS = {
    "OnlyBubbles Val": "./data/kaleel_final_dataset_val_OnlyBubbles_Grayscale.pth",
    "Combined Val": "./data/kaleel_final_dataset_val_Combined_Grayscale.pth",
}

TRAIN_DATASETS = {
    "OnlyBubbles Train": "./data/kaleel_final_dataset_train_OnlyBubbles_Grayscale.pth",
    "Combined Train": "./data/kaleel_final_dataset_train_Combined_Grayscale.pth",
}

DATASETS = {**VAL_DATASETS, **TRAIN_DATASETS}

# ------------------------ Scanned Bubble Datasets (Post Print-Scan) ------------------------------
SCANNED_DATASETS_DIR = "./data"

SCANNED_DATASETS = {
    "resnet20_combined": f"{SCANNED_DATASETS_DIR}/ResNet-Combined_correct_samples_1000_scanned_bubbles.pt",
    "cait_combined": f"{SCANNED_DATASETS_DIR}/CaiT-Combined_correct_samples_1000_scanned_bubbles.pt",
    "vgg16_combined": f"{SCANNED_DATASETS_DIR}/VGG-Combined_correct_samples_1000_scanned_bubbles.pt",
    "svm_combined": f"{SCANNED_DATASETS_DIR}/SVM-Combined_correct_samples_1000_scanned_bubbles.pt",
    "snn_vgg16_combined": f"{SCANNED_DATASETS_DIR}/SNN-VGG-Combined_correct_samples_1000_scanned_bubbles.pt",
    "snn_resnet20_combined": f"{SCANNED_DATASETS_DIR}/SNN-ResNet-Combined_correct_samples_1000_scanned_bubbles.pt",
    "expv2_vgg16": f"{SCANNED_DATASETS_DIR}/xAI_VGG16-Combined_correct_samples_1000_scanned_bubbles.pt",
    "expv2_resnet20": f"{SCANNED_DATASETS_DIR}/xAI-ResNet20-Combined_correct_samples_1000_scanned_bubbles.pt",
    "mambavision_combined": f"{SCANNED_DATASETS_DIR}/MambaVision-L2-Combined_correct_samples_1000_scanned_bubbles.pt",
}

# ========================================================================================
# Individual Experiments (Without UNet) - Uses Original Validation Data
# ========================================================================================

EXPERIMENTS_RESNET20 = {
    "resnet20_combined": {
        "ckpt_path": CHECKPOINTS["resnet20_combined"],
        "dataset_path": VAL_DATASETS["OnlyBubbles Val"],
        "use_unet": False,
    },
}

EXPERIMENTS_VGG16 = {
    "vgg16_combined": {
        "ckpt_path": CHECKPOINTS["vgg16_combined"],
        "dataset_path": VAL_DATASETS["OnlyBubbles Val"],
        "use_unet": False,
    },
}

EXPERIMENTS_SVM = {
    "svm_combined": {
        "ckpt_path": CHECKPOINTS["svm_combined"],
        "dataset_path": VAL_DATASETS["OnlyBubbles Val"],
        "use_unet": False,
    },
}

EXPERIMENTS_CAIT = {
    "cait_combined": {
        "ckpt_path": CHECKPOINTS["cait_combined"],
        "dataset_path": VAL_DATASETS["OnlyBubbles Val"],
        "use_unet": False,
    },
}

EXPERIMENTS_SNN_VGG16 = {
    "snn_vgg16_combined": {
        "ckpt_path": CHECKPOINTS["snn_vgg16_combined"],
        "dataset_path": VAL_DATASETS["OnlyBubbles Val"],
        "use_unet": False,
    },
}

EXPERIMENTS_SNN_RESNET20 = {
    "snn_resnet20_combined": {
        "ckpt_path": CHECKPOINTS["snn_resnet20_combined"],
        "dataset_path": VAL_DATASETS["OnlyBubbles Val"],
        "use_unet": False,
    },
}

EXPERIMENTS_XAI_RESNET20 = {
    "expv2_resnet20": {
        "ckpt_path": CHECKPOINTS["expv2_resnet20"],
        "dataset_path": VAL_DATASETS["OnlyBubbles Val"],
        "use_unet": False,
    },
}

EXPERIMENTS_XAI_VGG16 = {
    "expv2_vgg16": {
        "ckpt_path": CHECKPOINTS["expv2_vgg16"],
        "dataset_path": VAL_DATASETS["OnlyBubbles Val"],
        "use_unet": False,
    },
}

EXPERIMENTS_MAMBAVISION = {
    "mambavision_combined": {
        "ckpt_path": CHECKPOINTS["mambavision_combined"],
        "dataset_path": VAL_DATASETS["OnlyBubbles Val"],
        "use_unet": False,
    },
}

# ========================================================================================
# Individual Experiments (With UNet) - Uses Scanned Bubble Data
# ========================================================================================

EXPERIMENTS_UNET_RESNET20 = {
    "resnet20_combined": {
        "ckpt_path": CHECKPOINTS["resnet20_combined"],
        "dataset_path": SCANNED_DATASETS["resnet20_combined"],
        "use_unet": True,
        "unet_ckpt": UNET_CHECKPOINT,
        "display_name": "ResNet20-C",
    },
}

EXPERIMENTS_UNET_VGG16 = {
    "vgg16_combined": {
        "ckpt_path": CHECKPOINTS["vgg16_combined"],
        "dataset_path": SCANNED_DATASETS["vgg16_combined"],
        "use_unet": True,
        "unet_ckpt": UNET_CHECKPOINT,
        "display_name": "VGG16-C",
    },
}

EXPERIMENTS_UNET_SVM = {
    "svm_combined": {
        "ckpt_path": CHECKPOINTS["svm_combined"],
        "dataset_path": SCANNED_DATASETS["svm_combined"],
        "use_unet": True,
        "unet_ckpt": UNET_CHECKPOINT,
        "display_name": "SVM-C",
    },
}

EXPERIMENTS_UNET_CAIT = {
    "cait_combined": {
        "ckpt_path": CHECKPOINTS["cait_combined"],
        "dataset_path": SCANNED_DATASETS["cait_combined"],
        "use_unet": True,
        "unet_ckpt": UNET_CHECKPOINT,
        "display_name": "CaiT-C",
    },
}

EXPERIMENTS_UNET_SNN_VGG16 = {
    "snn_vgg16_combined": {
        "ckpt_path": CHECKPOINTS["snn_vgg16_combined"],
        "dataset_path": SCANNED_DATASETS["snn_vgg16_combined"],
        "use_unet": True,
        "unet_ckpt": UNET_CHECKPOINT,
        "display_name": "SNN_VGG16-C",
    },
}

EXPERIMENTS_UNET_SNN_RESNET20 = {
    "snn_resnet20_combined": {
        "ckpt_path": CHECKPOINTS["snn_resnet20_combined"],
        "dataset_path": SCANNED_DATASETS["snn_resnet20_combined"],
        "use_unet": True,
        "unet_ckpt": UNET_CHECKPOINT,
        "display_name": "SNN_ResNet20-C",
    },
}

EXPERIMENTS_UNET_XAI_RESNET20 = {
    "expv2_resnet20": {
        "ckpt_path": CHECKPOINTS["expv2_resnet20"],
        "dataset_path": SCANNED_DATASETS["expv2_resnet20"],
        "use_unet": True,
        "unet_ckpt": UNET_CHECKPOINT,
        "display_name": "Explainable_AI_ResNet20-C",
    },
}

EXPERIMENTS_UNET_XAI_VGG16 = {
    "expv2_vgg16": {
        "ckpt_path": CHECKPOINTS["expv2_vgg16"],
        "dataset_path": SCANNED_DATASETS["expv2_vgg16"],
        "use_unet": True,
        "unet_ckpt": UNET_CHECKPOINT,
        "display_name": "Explainable_AI_VGG16-C",
    },
}

EXPERIMENTS_UNET_MAMBAVISION = {
    "mambavision_combined": {
        "ckpt_path": CHECKPOINTS["mambavision_combined"],
        "dataset_path": SCANNED_DATASETS["mambavision_combined"],
        "use_unet": True,
        "unet_ckpt": UNET_CHECKPOINT,
        "display_name": "MambaVision-L2-C",
    },
}

# ========================================================================================
# All Models Combined (Without UNet) - Uses Original Validation Data
# ========================================================================================

EXPERIMENTS_ALL = {
    "resnet20_combined": {
        "ckpt_path": CHECKPOINTS["resnet20_combined"],
        "dataset_path": VAL_DATASETS["OnlyBubbles Val"],
        "use_unet": False,
    },
    "cait_combined": {
        "ckpt_path": CHECKPOINTS["cait_combined"],
        "dataset_path": VAL_DATASETS["OnlyBubbles Val"],
        "use_unet": False,
    },
    "vgg16_combined": {
        "ckpt_path": CHECKPOINTS["vgg16_combined"],
        "dataset_path": VAL_DATASETS["OnlyBubbles Val"],
        "use_unet": False,
    },
    "svm_combined": {
        "ckpt_path": CHECKPOINTS["svm_combined"],
        "dataset_path": VAL_DATASETS["OnlyBubbles Val"],
        "use_unet": False,
    },
    "snn_vgg16_combined": {
        "ckpt_path": CHECKPOINTS["snn_vgg16_combined"],
        "dataset_path": VAL_DATASETS["OnlyBubbles Val"],
        "use_unet": False,
    },
    "snn_resnet20_combined": {
        "ckpt_path": CHECKPOINTS["snn_resnet20_combined"],
        "dataset_path": VAL_DATASETS["OnlyBubbles Val"],
        "use_unet": False,
    },
    "expv2_vgg16": {
        "ckpt_path": CHECKPOINTS["expv2_vgg16"],
        "dataset_path": VAL_DATASETS["OnlyBubbles Val"],
        "use_unet": False,
    },
    "expv2_resnet20": {
        "ckpt_path": CHECKPOINTS["expv2_resnet20"],
        "dataset_path": VAL_DATASETS["OnlyBubbles Val"],
        "use_unet": False,
    },
    "mambavision_combined": {
        "ckpt_path": CHECKPOINTS["mambavision_combined"],
        "dataset_path": VAL_DATASETS["OnlyBubbles Val"],
        "use_unet": False,
    },
}

# ========================================================================================
# All Models Combined (With UNet) - Uses Scanned Bubble Data
# ========================================================================================

EXPERIMENTS_UNET_ALL = {
    "resnet20_combined": {
        "ckpt_path": CHECKPOINTS["resnet20_combined"],
        "dataset_path": SCANNED_DATASETS["resnet20_combined"],
        "use_unet": True,
        "unet_ckpt": UNET_CHECKPOINT,
        "display_name": "ResNet20-C",
    },
    "cait_combined": {
        "ckpt_path": CHECKPOINTS["cait_combined"],
        "dataset_path": SCANNED_DATASETS["cait_combined"],
        "use_unet": True,
        "unet_ckpt": UNET_CHECKPOINT,
        "display_name": "CaiT-C",
    },
    "vgg16_combined": {
        "ckpt_path": CHECKPOINTS["vgg16_combined"],
        "dataset_path": SCANNED_DATASETS["vgg16_combined"],
        "use_unet": True,
        "unet_ckpt": UNET_CHECKPOINT,
        "display_name": "VGG16-C",
    },
    "svm_combined": {
        "ckpt_path": CHECKPOINTS["svm_combined"],
        "dataset_path": SCANNED_DATASETS["svm_combined"],
        "use_unet": True,
        "unet_ckpt": UNET_CHECKPOINT,
        "display_name": "SVM-C",
    },
    "snn_vgg16_combined": {
        "ckpt_path": CHECKPOINTS["snn_vgg16_combined"],
        "dataset_path": SCANNED_DATASETS["snn_vgg16_combined"],
        "use_unet": True,
        "unet_ckpt": UNET_CHECKPOINT,
        "display_name": "SNN_VGG16-C",
    },
    "snn_resnet20_combined": {
        "ckpt_path": CHECKPOINTS["snn_resnet20_combined"],
        "dataset_path": SCANNED_DATASETS["snn_resnet20_combined"],
        "use_unet": True,
        "unet_ckpt": UNET_CHECKPOINT,
        "display_name": "SNN_ResNet20-C",
    },
    "expv2_vgg16": {
        "ckpt_path": CHECKPOINTS["expv2_vgg16"],
        "dataset_path": SCANNED_DATASETS["expv2_vgg16"],
        "use_unet": True,
        "unet_ckpt": UNET_CHECKPOINT,
        "display_name": "Explainable_AI_VGG16-C",
    },
    "expv2_resnet20": {
        "ckpt_path": CHECKPOINTS["expv2_resnet20"],
        "dataset_path": SCANNED_DATASETS["expv2_resnet20"],
        "use_unet": True,
        "unet_ckpt": UNET_CHECKPOINT,
        "display_name": "Explainable_AI_ResNet20-C",
    },
    "mambavision_combined": {
        "ckpt_path": CHECKPOINTS["mambavision_combined"],
        "dataset_path": SCANNED_DATASETS["mambavision_combined"],
        "use_unet": True,
        "unet_ckpt": UNET_CHECKPOINT,
        "display_name": "MambaVision-L2-C",
    },
}
</file>

<file path="README.md">
Reference:

Nicolas Papernot, Patrick D. McDaniel, Ian J. Goodfellow, Somesh Jha, Z. Berkay Celik, and Ananthram Swami. Practical Black-Box Attacks against Machine Learning. In ACM AsiaCCS 2017, pages 506–519, 2017.

Macas, Mayra, Chunming Wu, and Walter Fuertes. "Adversarial examples: A survey of attacks and defenses in deep learning-enabled cybersecurity systems." Expert Systems with Applications 238 (2024): 122223.
</file>

<file path="checkpoint/.gitkeep">

</file>

<file path="data/.gitkeep">

</file>

<file path="model_architecture/cait.py">
# https://github.com/lucidrains/vit-pytorch/blob/main/vit_pytorch/cait.py

from random import randrange
import torch
from torch import nn, einsum
import torch.nn.functional as F

from einops import rearrange, repeat
from einops.layers.torch import Rearrange

# helpers

def exists(val):
    return val is not None

def dropout_layers(layers, dropout):
    if dropout == 0:
        return layers

    num_layers = len(layers)
    to_drop = torch.zeros(num_layers).uniform_(0., 1.) < dropout

    # make sure at least one layer makes it
    if all(to_drop):
        rand_index = randrange(num_layers)
        to_drop[rand_index] = False

    layers = [layer for (layer, drop) in zip(layers, to_drop) if not drop]
    return layers

# classes

class LayerScale(nn.Module):
    def __init__(self, dim, fn, depth):
        super().__init__()
        if depth <= 18:  # epsilon detailed in section 2 of paper
            init_eps = 0.1
        elif depth > 18 and depth <= 24:
            init_eps = 1e-5
        else:
            init_eps = 1e-6

        scale = torch.zeros(1, 1, dim).fill_(init_eps)
        self.scale = nn.Parameter(scale)
        self.fn = fn
    def forward(self, x, **kwargs):
        return self.fn(x, **kwargs) * self.scale

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn
    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        return self.net(x)

class Attention(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0.):
        super().__init__()
        inner_dim = dim_head *  heads
        self.heads = heads
        self.scale = dim_head ** -0.5

        self.to_q = nn.Linear(dim, inner_dim, bias = False)
        self.to_kv = nn.Linear(dim, inner_dim * 2, bias = False)

        self.attend = nn.Softmax(dim = -1)

        self.mix_heads_pre_attn = nn.Parameter(torch.randn(heads, heads))
        self.mix_heads_post_attn = nn.Parameter(torch.randn(heads, heads))

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x, context = None):
        b, n, _, h = *x.shape, self.heads

        context = x if not exists(context) else torch.cat((x, context), dim = 1)

        qkv = (self.to_q(x), *self.to_kv(context).chunk(2, dim = -1))
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = h), qkv)

        dots = einsum('b h i d, b h j d -> b h i j', q, k) * self.scale

        dots = einsum('b h i j, h g -> b g i j', dots, self.mix_heads_pre_attn)    # talking heads, pre-softmax
        attn = self.attend(dots)
        attn = einsum('b h i j, h g -> b g i j', attn, self.mix_heads_post_attn)   # talking heads, post-softmax

        out = einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)

class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout = 0., layer_dropout = 0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        self.layer_dropout = layer_dropout

        for ind in range(depth):
            self.layers.append(nn.ModuleList([
                LayerScale(dim, PreNorm(dim, Attention(dim, heads = heads, dim_head = dim_head, dropout = dropout)), depth = ind + 1),
                LayerScale(dim, PreNorm(dim, FeedForward(dim, mlp_dim, dropout = dropout)), depth = ind + 1)
            ]))
    def forward(self, x, context = None):
        layers = dropout_layers(self.layers, dropout = self.layer_dropout)

        for attn, ff in layers:
            x = attn(x, context = context) + x
            x = ff(x) + x
        return x

class CaiT(nn.Module):
    def __init__(
        self,
        *,
        image_size,
        patch_size,
        num_classes,
        num_channels=3,
        dim,
        depth,
        cls_depth,
        heads,
        mlp_dim,
        dim_head = 64,
        dropout = 0.,
        emb_dropout = 0.,
        layer_dropout = 0.
    ):
        super().__init__()
        #assert image_size % patch_size == 0, 'Image dimensions must be divisible by the patch size.'
        self.image_size = image_size
        num_patches = (image_size[0] * image_size[1])// patch_size**2
        patch_dim = num_channels * patch_size ** 2 # for color use 3 * patch_size ** 2

        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1 = patch_size, p2 = patch_size),
            nn.Linear(patch_dim, dim),
        )

        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches, dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))

        self.dropout = nn.Dropout(emb_dropout)

        self.patch_transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout, layer_dropout)
        self.cls_transformer = Transformer(dim, cls_depth, heads, dim_head, mlp_dim, dropout, layer_dropout)

        self.mlp_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, num_classes)
        )

    def forward(self, img):
        x = self.to_patch_embedding(img)
        b, n, _ = x.shape

        x += self.pos_embedding[:, :n]
        x = self.dropout(x)

        x = self.patch_transformer(x)

        cls_tokens = repeat(self.cls_token, '() n d -> b n d', b = b)
        x = self.cls_transformer(cls_tokens, context = x)

        return self.mlp_head(x[:, 0])
</file>

<file path="model_architecture/CarliniNetwork.py">
# Network constructors for the adaptive black-box attack 
# Modified for grayscale voter data (1 channel, 40x50)
import torch.nn
import torch.nn.functional as F


class CarliniNetwork(torch.nn.Module):
    def __init__(self, imgH=40, imgW=50, numChannels=1, numClasses=2):
        """
        Carlini Network adapted for grayscale voter data
        """
        super(CarliniNetwork, self).__init__()
        
        # Parameters for the network 
        params = [64, 64, 128, 128, 256, 256]
        
        # Store image dimensions
        self.imgH = imgH
        self.imgW = imgW
        self.numChannels = numChannels
        
        # Create the layers
        # Conv2D(params[0], (3, 3), input_shape=inputShape) + Activation('relu')
        self.conv0 = torch.nn.Conv2d(in_channels=numChannels, out_channels=params[0], kernel_size=(3,3), stride=1)
        
        # Conv2D(params[1], (3, 3)) + Activation('relu')
        self.conv1 = torch.nn.Conv2d(in_channels=params[0], out_channels=params[1], kernel_size=(3,3), stride=1)
        
        # MaxPooling2D(pool_size=(2, 2))
        self.mp0 = torch.nn.MaxPool2d(kernel_size=(2,2))
        
        # Conv2D(params[2], (3, 3)) + Activation('relu')
        self.conv2 = torch.nn.Conv2d(in_channels=params[1], out_channels=params[2], kernel_size=(3,3), stride=1)
        
        # Conv2D(params[3], (3, 3)) + Activation('relu')
        self.conv3 = torch.nn.Conv2d(in_channels=params[2], out_channels=params[3], kernel_size=(3,3), stride=1)
        
        # MaxPooling2D(pool_size=(2, 2))
        self.mp1 = torch.nn.MaxPool2d(kernel_size=(2,2))
        
        # Compute flatten size dynamically
        testInput = torch.zeros((1, numChannels, imgH, imgW))
        outputShape = self.figureOutFlattenShape(testInput)
        
        # Dense(params[4]) + Activation('relu')
        self.forward0 = torch.nn.Linear(in_features=outputShape[1], out_features=params[4])
        
        # Dropout(0.5)
        self.drop0 = torch.nn.Dropout(0.5)
        
        # Dense(params[5]) + Activation('relu')
        self.forward1 = torch.nn.Linear(in_features=params[4], out_features=params[5])
        
        # Dense(numClasses) + Activation('softmax')
        self.forward2 = torch.nn.Linear(in_features=params[5], out_features=numClasses)

    def forward(self, x):
        out = F.relu(self.conv0(x))
        out = F.relu(self.conv1(out))
        out = self.mp0(out)
        out = F.relu(self.conv2(out))
        out = F.relu(self.conv3(out))
        out = self.mp1(out)
        out = out.view(out.size(0), -1)  # Flatten
        out = F.relu(self.forward0(out))
        out = self.drop0(out)
        out = F.relu(self.forward1(out))
        out = F.softmax(self.forward2(out), dim=1)
        return out

    def figureOutFlattenShape(self, x):
        """Compute the flatten shape after conv layers"""
        out = F.relu(self.conv0(x))
        out = F.relu(self.conv1(out))
        out = self.mp0(out)
        out = F.relu(self.conv2(out))
        out = F.relu(self.conv3(out))
        out = self.mp1(out)
        out = out.view(out.size(0), -1)
        return out.shape
</file>

<file path="model_architecture/MultiOutputSVM.py">
import torch
import torch.nn as nn
import torch.nn.functional as F

class BaseSVM(nn.Module):
    """Single-logit linear model: margin f(x)=w^T x + b"""
    def __init__(self, input_dim):
        super().__init__()
        self.linear = nn.Linear(input_dim, 1, bias=True)
    def forward(self, x):
        x = x.view(x.size(0), -1)       # [B, 2000]
        return self.linear(x).squeeze(1)

class MultiOutputSVM(nn.Module):
    """Two-logit head: logits=[-f, +f] (symmetric, no bias shift)."""
    def __init__(self, input_dim, base_state_dict):
        super().__init__()
        self.linear = nn.Linear(input_dim, 1, bias=True)
        self.out    = nn.Linear(1, 2, bias=True)
        self.load_base_and_fix_head(base_state_dict)

    def load_base_and_fix_head(self, state_dict):
        # copy base weights (w,b) into the first linear
        base = BaseSVM(self.linear.in_features)
        base.load_state_dict(state_dict)
        with torch.no_grad():
            self.linear.weight.copy_(base.linear.weight)
            self.linear.bias.copy_(base.linear.bias)
            # symmetric mapping to two logits
            self.out.weight.zero_(); self.out.bias.zero_()
            self.out.weight[0,0] = -1.0   # class 0 logit = -margin
            self.out.weight[1,0] =  1.0   # class 1 logit = +margin

    def forward(self, x):
        f = self.linear(x.view(x.size(0), -1)).squeeze(1)   # margin
        logits = self.out(f.unsqueeze(1))                   # [B,2]
        return logits   # (use CE or softmax outside as needed)
</file>

<file path="model_architecture/ResNet.py">
#This is Pytorch version of the ResNet V2 architecture copied over from Keras 
#Reference:
#[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
#    Deep Residual Learning for Image Recognition. arXiv:1512.03385
#[2] https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py
#If you use this implementation in you work, please don't forget to mention the
#author, Yerlan Idelbayev.

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from torch.autograd import Variable
import numpy

#__all__ = ['ResNet', 'resnet20', 'resnet32', 'resnet44', 'resnet56', 'resnet110', 'resnet1202']

def _weights_init(m):
    classname = m.__class__.__name__
    #print(classname)
    if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
        init.kaiming_normal_(m.weight)

class LambdaLayer(nn.Module):
    def __init__(self, lambd):
        super(LambdaLayer, self).__init__()
        self.lambd = lambd

    def forward(self, x):
        return self.lambd(x)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, res_block, activation, batch_normalization, in_planes, planes, stride):
        super(BasicBlock, self).__init__()
        self.res_block = res_block
        self.activation = activation 
        self.batch_normalization = batch_normalization

        #Keras ResNetV2 architecture
        if res_block == 0:
            self.bn1 = nn.BatchNorm2d(in_planes)
            self.conv1 = nn.Conv2d(in_planes, in_planes, kernel_size=1, stride=stride, bias=True)
        else:
            self.bn1 = nn.BatchNorm2d(planes)
            self.conv1 = nn.Conv2d(planes, in_planes, kernel_size=1, stride=stride, bias=True)

        self.bn2 = nn.BatchNorm2d(in_planes)
        #self.conv2 = nn.Conv2d(in_planes, in_planes, kernel_size=3, stride=1, padding=1, bias=True)
        self.conv2 = nn.Conv2d(in_planes, in_planes, kernel_size=3, stride=1, padding=1, bias=True)

        self.bn3 = nn.BatchNorm2d(in_planes)
        self.conv3 = nn.Conv2d(in_planes, planes, kernel_size=1, stride=1, bias=True)

        #Skip connection setup here I think 
        self.shortcut = nn.Sequential()
        if res_block == 0:
            self.shortcut = nn.Sequential(nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride, bias=True))

    def forward(self, x):
        #The forward is slightly tricky to copy Keras implementation 
        if self.activation==True and self.batch_normalization ==True: #Fire everything 
            out =  self.conv1(F.relu(self.bn1(x))) #BN->RELU->CONV
        elif self.activation==True and self.batch_normalization ==False: #Activation, no batch normalization 
            out =  self.conv1(F.relu(x))
        elif self.activation==False and self.batch_normalization ==True: #No activation, batch normalization 
            out =  self.conv1(self.bn1(x))
        elif self.activation==False and self.batch_normalization ==False: #No activation, no batch normalization 
            out =  self.conv1(x)
        out =  self.conv2(F.relu(self.bn2(out)))
        out =  self.conv3(F.relu(self.bn3(out)))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, inputShape, dropOutRate = 0, numClasses=10):
        super(ResNet, self).__init__()
        self.numClasses = numClasses
        self.dropOutRate = dropOutRate
        self.in_planes = 16
        self.conv1 = nn.Conv2d(inputShape[1], 16, kernel_size=3, stride=1, padding=1, bias=True)
        #print("WARNING: NETWORK ONLY CONFIGURED FOR GRAYSCALE.")
        #self.conv1 = nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1, bias=True)

        self.bn1 = nn.BatchNorm2d(16)
        #In PyTorch these are called layers, in Keras they are called stages the ResNets all have 3 stacks 
        #Each stack contains a certain number of blocks (6 in the ResNet56 implementation)
        #The first stack 
        stageNum = 0
        in_planes = 16
        self.layer1 = self._make_layer(stageNum, block, in_planes, num_blocks[0])
        #The second stack
        stageNum = 1
        in_planes = 64
        self.layer2 = self._make_layer(stageNum, block, in_planes, num_blocks[1])
        #The third stack
        stageNum = 2
        in_planes = 128
        self.layer3 = self._make_layer(stageNum, block, in_planes, num_blocks[2])

        #Classifer 
        classifierInputSize = in_planes * 2
        self.bn2 = nn.BatchNorm2d(classifierInputSize) #x = BatchNormalization()(x)
        
        #breaker = 5
        #if inputImageSize == 32 and num_blocks[0]==6:
        #    forwardInputSize = 256
        #elif inputImageSize ==224 and num_blocks[0]==6:
        #    forwardInputSize =12544
        #elif inputImageSize == 32 and num_blocks[0]==18:
        #    forwardInputSize =256 
        #else:
        #    raise ValueError("Input size not configured for the architecture. Compute the forward input size and recode around line 105.")
        #zz = torch.zeros(inputShape)
        #print(zz.shape)

        forwardInputSize = self.forwardDebug(torch.zeros(inputShape))
        self.sm = nn.Linear(in_features=forwardInputSize, out_features=numClasses)
        self.apply(_weights_init)
        
        # Sigmoid output replaces softmax
        #self.sigmoid = nn.Sigmoid()

        # Dropout layer
        self.drop = nn.Dropout(p = self.dropOutRate)

    def _make_layer(self, stageNum, block, in_planes, num_blocks):
        layers = []
        for res_block in range(0, num_blocks):  
            #This setup is almost all directly copied from Keras 
            activation = True
            batch_normalization = True
            strides = 1
            if stageNum == 0:
                planes = in_planes * 4
                if res_block == 0:  # first layer and first stage
                    activation = False
                    batch_normalization = False
            else:
                planes = in_planes * 2
                if res_block == 0:  # first layer but not first stage
                    strides = 2    # downsample
            #End of Keras parameter setup 
            layers.append(block(res_block, activation, batch_normalization, in_planes, planes, strides))
        return nn.Sequential(*layers)

    #Should ONLY be used for debugging purposes
    def forwardDebug(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.bn2(out)
        out = F.avg_pool2d(out, 8) #pool size 8
        out = out.view(out.size(0), -1) #This should replicate behavior of flatten
        return out.shape[1]

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.bn2(out)
        out = F.avg_pool2d(out, 8) #pool size 8
        out = out.view(out.size(0), -1) #This should replicate behavior of flatten
        if self.dropOutRate != 0: out = self.drop(out)
        #out = self.sm(out)
        if self.numClasses == 1: 
            out = torch.sigmoid(out)
            #out = self.sigmoid(out)
        #out = F.softmax(self.sm(out))
        out = self.sm(out)
        return out

def resnet20(inputShape, dropOutRate, numClasses):
    return ResNet(BasicBlock, [2, 2, 2], inputShape, dropOutRate, numClasses)

def resnet56(inputShape, dropOutRate, numClasses):
    return ResNet(BasicBlock, [6, 6, 6], inputShape, dropOutRate, numClasses) #This is V2
    #return ResNet(BasicBlock, [9, 9, 9]) #This was V1 in Keras kind of

def resnet164(inputImageSize, dropOutRate, numClasses):
    return ResNet(BasicBlock, [18, 18, 18], inputImageSize, dropOutRate, numClasses)

def resnet1001(inputImageSize, dropOutRate, numClasses):
    return ResNet(BasicBlock, [111, 111, 111], inputImageSize, dropOutRate, numClasses)
</file>

<file path="model_architecture/VGG.py">
#From: https://github.com/kuangliu/pytorch-cifar
import torch
import torch.nn as nn

cfg = {
    'VGG11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'VGG19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}  


class VGG(nn.Module):
    def __init__(self, vgg_name, imgH, imgW, numClasses):
        super(VGG, self).__init__()
        self.features = self._make_layers(cfg[vgg_name])
        
        #Figure out the shape for flatten
        #Changed to be grayscale originally was
        #x = torch.zeros(1,3,imgH, imgW)  
        x = torch.zeros(1,1,imgH, imgW)  

        outputShape = self.FigureOutFlattenShape(x)
        self.classifier = nn.Linear(outputShape[1], numClasses)
        #self.classifier = nn.Linear(512, 10)

    #Added by K to figure out the flattening layer 
    def FigureOutFlattenShape(self, x):
        out = self.features(x)
        out = out.view(out.size(0), -1)
        return out.shape

    def forward(self, x):
        out = self.features(x)
        out = out.view(out.size(0), -1)
        out = self.classifier(out)
        return out

    def _make_layers(self, cfg):
        layers = []
        #in_channels = 3 #changed for grayscale
        in_channels = 1
        for x in cfg:
            if x == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                layers += [nn.Conv2d(in_channels, x, kernel_size=3, padding=1),
                           nn.BatchNorm2d(x),
                           nn.ReLU(inplace=True)]
                in_channels = x
        layers += [nn.AvgPool2d(kernel_size=1, stride=1)]
        return nn.Sequential(*layers)


def test():
    net = VGG('VGG11')
    x = torch.randn(2,3,32,32)
    y = net(x)
    print(y.size())
</file>

<file path="DataLoaderGiant.py">
#Dataloader giant combines multiple dataloaders and only loads them into RAM when needed 
import utils as DMP
import torch 
import numpy

class DataLoaderGiant():
    def __init__(self, homeDir, batchSize):
        self.homeDir = homeDir #This is where all the dataloaders will be saved 
        self.dataLoaderDirList = [] #List to hold the names of the dataloaders 
        self.batchSize = batchSize

    #Add a dataloader to the directory 
    def AddLoader(self, dataLoaderName, dataLoader):
        #Torch limits the amount of data we can save to disk so we must use numpy to save 
        #torch.save(dataLoader, self.homeDir+dataLoaderName)
        #First convert the tensor to a dataloader 
        xDataPytorch, yDataPytorch = DMP.DataLoaderToTensor(dataLoader)
        #Second conver the pytorch arrays to numpy arrays for saving 
        xDataNumpy = xDataPytorch.cpu().detach().numpy()
        yDataNumpy = yDataPytorch.cpu().detach().numpy()
        #Save the data using numpy
        numpy.save(self.homeDir+dataLoaderName+"XData", xDataNumpy)
        numpy.save(self.homeDir+dataLoaderName+"YData", yDataNumpy)
        #Save the file location string so we can re-load later 
        self.dataLoaderDirList.append(dataLoaderName)
        #Delete the dataloader and associated variables from memory 
        del dataLoader
        del xDataPytorch
        del yDataPytorch
        del xDataNumpy
        del yDataNumpy
       
    def GetLoaderAtIndex(self, index):
        currentDataLoaderDir = self.homeDir + self.dataLoaderDirList[index]
        #First load the numpy arrays 
        xData = numpy.load(currentDataLoaderDir+"XData.npy")
        yData = numpy.load(currentDataLoaderDir+"YData.npy")
        #Create a dataloader 
        currentDataLoader = DMP.TensorToDataLoader(torch.from_numpy(xData), torch.from_numpy(yData), transforms = None, batchSize = self.batchSize, randomizer = None)
        #currentDataLoader = torch.load(currentDataLoaderDir)
        #Do some memory clean up
        del xData
        del yData
        return currentDataLoader

    def GetNumberOfLoaders(self):
        return len(self.dataLoaderDirList)
</file>

<file path=".gitignore">
# ============================================
# EXPERIMENT OUTPUT FOLDERS
# ============================================
# Date-named experiment folders (e.g., February-20-2026, Adaptive Attack)
*-20[0-9][0-9],*/
*-Adaptive Attack*/
*Adaptive Attack*/

Others/

# ============================================
# CHECKPOINT FILES
# ============================================
checkpoint/*
!checkpoint/.gitkeep

# ============================================
# DATA FILES
# ============================================
data/*
!data/.gitkeep

# ============================================
# NUMPY & PYTORCH FILES
# ============================================
*.npy
*.npz
*.pth
*.pt
*.th

# Model files without extensions
SyntheticModel
AdvLoaderAPGD
**/SyntheticModel
**/AdvLoaderAPGD

# ============================================
# PYTHON
# ============================================
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
wheels/
*.egg-info/
.installed.cfg
*.egg
MANIFEST

# ============================================
# VIRTUAL ENVIRONMENT
# ============================================
venv/
env/
ENV/
.venv/
.env/

# ============================================
# IDE & EDITORS
# ============================================
.vscode/
.idea/
*.swp
*.swo
*~
*.sublime-project
*.sublime-workspace

# ============================================
# OS GENERATED FILES
# ============================================
.DS_Store
.DS_Store?
._*
.Spotlight-V100
.Trashes
ehthumbs.db
Thumbs.db
desktop.ini

# ============================================
# JUPYTER NOTEBOOK
# ============================================
.ipynb_checkpoints/
*.ipynb_checkpoints

# ============================================
# SKLEARN/JOBLIB FILES
# ============================================
*.joblib

# ============================================
# TEMPORARY & MISC FILES
# ============================================
*.tmp
*.temp
*.bak
untitled.txt
</file>

<file path="ModelFactory.py">
import torch
import torch.nn as nn

from typing import Union, List, Tuple, Optional
import sys
from pathlib import Path

from spikingjelly.activation_based import surrogate, neuron, functional

from model_architecture import ResNet, cait, VGG, MultiOutputSVM
from model_architecture.UNet import UNet
from model_architecture.spiking_vgg_voter import spiking_vgg16_bn_voter
from model_architecture.spiking_resnet_voter import spiking_resnet20_voter

# ----------- Wrapper Class ---------------------
class SNNWrapper(nn.Module):
    """
    Wrapper that handles time dimension internally.
    Allows using existing utils.py functions without modification.
    """
    
    def __init__(self, snn_model, T=4):
        super(SNNWrapper, self).__init__()
        self.snn = snn_model
        self.T = T
    
    def forward(self, x):
        # Add time dimension: [N, C, H, W] → [T, N, C, H, W]
        x_seq = x.unsqueeze(0).repeat(self.T, 1, 1, 1, 1)
        
        # Forward through SNN
        out_seq = self.snn(x_seq)
        
        # Average over time
        out = out_seq.mean(0)
        
        # Reset membrane
        functional.reset_net(self.snn)
        
        return out

class LogitsOnlyWrapper(nn.Module):
    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model

    def forward(self, x):
        output = self.model(x)
        if isinstance(output, dict):
            if "logits" in output:
                return output["logits"]
            raise KeyError("Model output dict missing 'logits' key.")
        if isinstance(output, tuple):
            return output[0]
        return output

class UNetModelWrapper(nn.Module):
    """
    Wrapper that passes input through UNet (denoiser) then through classifier model.
    
    Flow: Input → UNet → Model → Output
    
    This allows attacking the combined UNet+Model as a single unit.
    """
    
    def __init__(self, unet: nn.Module, model: nn.Module):
        super().__init__()
        self.unet = unet
        self.model = model
    
    def forward(self, x):
        # Pass through UNet (denoiser)
        cleaned = self.unet(x)
        
        # Handle if UNet returns tuple/list
        if isinstance(cleaned, (tuple, list)):
            cleaned = cleaned[0]
        
        # Clamp to valid range
        cleaned = cleaned.clamp(0.0, 1.0)
        
        # Pass through classifier
        output = self.model(cleaned)
        
        return output

# ----------------- MODEL FACTORY -----------------
class ModelFactory:
    def __init__(self, device: Optional[torch.device] = None):
        self.device = (
            device
            if device
            else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        )
        # Base directory for resolving paths
        self.base_dir = Path(__file__).resolve().parent

    def get_model(
        self,
        model_name: str,
        checkpoint_path: Union[str, List[str]] = None,
    ) -> nn.Module:
        model_name = model_name.lower()

        if "unet" in model_name and "+" not in model_name:
            # Pure UNet model
            return self._create_unet(checkpoint_path)
        elif "expv2" in model_name or "explainable" in model_name:
            return self._create_ppnet_v2_direct(
                checkpoint_path,
            )
        elif "snn_resnet" in model_name or "snn-resnet" in model_name:
            return self._create_snn_resnet(
                checkpoint_path,
            )
        elif "snn_vgg" in model_name or "snn-vgg" in model_name:
            return self._create_snn_vgg(
                checkpoint_path,
            )
        elif "carlini" in model_name:
            return self._create_carlini(
                checkpoint_path
            )
        elif "resnet" in model_name:
            return self._create_resnet(
                checkpoint_path,
            )
        elif "cait" in model_name:
            return self._create_cait(
                checkpoint_path,
            )
        elif "vgg11" in model_name:
            # VGG11 must come before vgg16 check
            return self._create_vgg11(
                checkpoint_path,
            )
        elif "vgg" in model_name:
            return self._create_vgg16(
                checkpoint_path,
            )
        elif "svm" in model_name:
            if isinstance(checkpoint_path, (list, tuple)) and len(checkpoint_path) == 2:
                return self._create_svm(checkpoint_path[0], checkpoint_path[1])
            else:
                raise ValueError(
                    "SVM requires a list/tuple of two paths: [base_path, multi_path]"
                )
        elif "mamba" in model_name or "mambavision" in model_name:
            return self._create_mambavision(
                checkpoint_path,
                )
        else:
            raise ValueError(f"Model '{model_name}' not recognized.")

    def get_unet_model_wrapper(
        self,
        model_name: str,
        model_checkpoint: Union[str, List[str]],
        unet_checkpoint: str,
    ) -> nn.Module:
        """
        Create a UNet+Model wrapper for defense evaluation.
        
        Args:
            model_name: Name of the classifier model
            model_checkpoint: Path to classifier checkpoint
            unet_checkpoint: Path to UNet checkpoint
            
        Returns:
            UNetModelWrapper that combines UNet denoiser with classifier
        """
        # Load UNet
        unet = self._create_unet(unet_checkpoint)
        
        # Load classifier model
        model = self.get_model(model_name, model_checkpoint)
        
        # Create wrapper
        wrapper = UNetModelWrapper(unet, model).to(self.device)
        wrapper.eval()
        
        return wrapper

    def _create_unet(self, checkpoint_path: str) -> nn.Module:
        """Load UNet autoencoder/denoiser model."""
        unet = UNet().to(self.device)
        
        ckpt = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        
        # Handle different checkpoint formats
        if isinstance(ckpt, dict) and "state_dict" in ckpt:
            state = ckpt["state_dict"]
        else:
            state = ckpt
        
        unet.load_state_dict(state)
        unet.eval()
        
        return unet


    def _create_resnet(
        self,
        checkpoint_path: str,
        input_size=[1, 1, 40, 50],
        num_classes=2,
        dropout=0.0,
    ) -> nn.Module:
        model = ResNet.resnet20(input_size, dropout, num_classes).to(self.device)
        checkpoint = torch.load(
            checkpoint_path, map_location=self.device, weights_only=False
        )
        model.load_state_dict(checkpoint["state_dict"])
        model.eval()
        return model

    def _create_cait(self, checkpoint_path: str, num_classes=2) -> nn.Module:
        model = cait.CaiT(
            image_size=(40, 50),
            patch_size=5,
            num_classes=num_classes,
            num_channels=1,
            dim=512,
            depth=16,
            cls_depth=2,
            heads=8,
            mlp_dim=2048,
            dropout=0.1,
            emb_dropout=0.1,
            layer_dropout=0.05,
        ).to(self.device)

        checkpoint = torch.load(
            checkpoint_path, map_location=self.device, weights_only=False
        )
        model.load_state_dict(checkpoint["state_dict"])

        if hasattr(model, "patch_transformer"):
            model.patch_transformer.layer_dropout = 0.0
        if hasattr(model, "cls_transformer"):
            model.cls_transformer.layer_dropout = 0.0

        model.eval()
        return model

    def _create_vgg11(
        self, 
        checkpoint_path: Optional[str] = None, 
        num_classes=2
    ) -> nn.Module:
        """Create VGG11 model (used as synthetic model for transfer attacks)."""
        model = VGG.VGG("VGG11", 40, 50, num_classes).to(self.device)
        
        if checkpoint_path is not None:
            raw = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
            state = raw.get("state_dict", raw)
            state = {(k[7:] if k.startswith("module.") else k): v for k, v in state.items()}
            model.load_state_dict(state, strict=False)
            model.eval()
        
        return model

    def _create_vgg16(self, checkpoint_path: str, num_classes=2) -> nn.Module:
        model = VGG.VGG("VGG16", 40, 50, num_classes).to(self.device)

        raw = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        state = raw.get("state_dict", raw)

        state = {(k[7:] if k.startswith("module.") else k): v for k, v in state.items()}

        model.load_state_dict(state, strict=False)
        model.eval()
        return model

    def _create_svm(self, base_path: str, multi_path: str) -> nn.Module:
        input_dim = 1 * 40 * 50
        base_state = torch.load(base_path, map_location="cpu", weights_only=False)

        model = MultiOutputSVM.MultiOutputSVM(input_dim, base_state).to(self.device)

        multi_state = torch.load(multi_path, map_location="cpu", weights_only=False)
        model.load_state_dict(multi_state)
        model.eval()
        return model

    def _create_carlini(
        self,
        checkpoint_path: Optional[str] = None,
        img_h: int = 40,
        img_w: int = 50,
        num_channels: int = 1,
        num_classes: int = 2,
    ) -> nn.Module:
        model = CarliniNetwork.CarliniNetwork(
            imgH=img_h,
            imgW=img_w,
            numChannels=num_channels,
            numClasses=num_classes,
        ).to(self.device)

        if checkpoint_path is not None:
            checkpoint = torch.load(
                checkpoint_path, map_location=self.device, weights_only=False
            )
            # Handle different checkpoint formats
            if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
                model.load_state_dict(checkpoint["state_dict"])
            else:
                model.load_state_dict(checkpoint)
            model.eval()

        return model

    def _create_snn_vgg(
        self,
        checkpoint_path: str,
        imgH: int = 40,
        imgW: int = 50,
        num_classes: int = 2,
        T: int = 4,
    ) -> nn.Module:
        """Create Spiking VGG16-BN model with SNNWrapper."""
        snn_model = spiking_vgg16_bn_voter(
            imgH=imgH,
            imgW=imgW,
            num_classes=num_classes,
            spiking_neuron=neuron.IFNode,
            surrogate_function=surrogate.ATan(),
            detach_reset=True,
        )
        functional.set_step_mode(snn_model, 'm')
        
        checkpoint = torch.load(
            checkpoint_path, map_location=self.device, weights_only=False
        )
        snn_model.load_state_dict(checkpoint['model'])
        
        model = SNNWrapper(snn_model, T=T)
        model = model.to(self.device)
        model.eval()
        return model

    def _create_snn_resnet(
        self,
        checkpoint_path: str,
        imgH: int = 40,
        imgW: int = 50,
        num_classes: int = 2,
        T: int = 4,
    ) -> nn.Module:
        """Create Spiking ResNet-20 model with SNNWrapper."""
        snn_model = spiking_resnet20_voter(
            imgH=imgH,
            imgW=imgW,
            num_classes=num_classes,
            spiking_neuron=neuron.IFNode,
            surrogate_function=surrogate.ATan(),
            detach_reset=True,
        )
        functional.set_step_mode(snn_model, 'm')
        
        checkpoint = torch.load(
            checkpoint_path, map_location=self.device, weights_only=False
        )
        snn_model.load_state_dict(checkpoint['model'])
        
        model = SNNWrapper(snn_model, T=T)
        model = model.to(self.device)
        model.eval()
        return model

    def _create_ppnet_v2_direct(self, checkpoint_path: str) -> nn.Module:
        """
        Load Explainable AI (ProtoPNet v2) model.
        
        Directory structure:
        Thesis/
        ├── Linf-BlackBoxAttack/
        │   └── ModelFactory.py       ← We are here (self.base_dir)
        └── Explanaible_AI/
            ├── cosine-is-almost/
            │   └── protopnext/
            │       └── protopnet/
            └── models/
                └── architecture/
                    └── ResNet.py
        """
        # ═══════════════════════════════════════════════════════════════
        # Path to Explanaible_AI directory (sibling of Linf-BlackBoxAttack)
        # ═══════════════════════════════════════════════════════════════
        
        # self.base_dir = Linf-BlackBoxAttack/
        # self.base_dir.parent = Thesis/
        # _EXPLAINABLE_DIR = Thesis/Explanaible_AI/
        
        _EXPLAINABLE_DIR = self.base_dir.parent / "Explanaible_AI"
        
        # Verify the directory exists
        if not _EXPLAINABLE_DIR.exists():
            raise FileNotFoundError(
                f"Explanaible_AI directory not found at: {_EXPLAINABLE_DIR}\n"
                f"Expected structure: {self.base_dir.parent}/Explanaible_AI/"
            )
        
        # Path to cosine-is-almost repo
        _COSINE_DIR = _EXPLAINABLE_DIR / "cosine-is-almost"
        _PPNEXT_DIR = _COSINE_DIR / "protopnext"
        
        # Path to base directory (for models.architecture.ResNet)
        _BASE_DIR = _EXPLAINABLE_DIR

        # ═══════════════════════════════════════════════════════════════
        # Add ALL required paths to sys.path
        # ═══════════════════════════════════════════════════════════════
        
        paths_to_add = [
            str(_PPNEXT_DIR),    # For: from protopnet.* import ...
            str(_COSINE_DIR),    # For: other cosine-is-almost imports
            str(_BASE_DIR),      # For: from models.architecture.ResNet import ...
        ]
        
        paths_added = []
        for p in paths_to_add:
            if p not in sys.path:
                sys.path.insert(0, p)
                paths_added.append(p)

        try:
            ppnet = torch.load(
                str(checkpoint_path), map_location=self.device, weights_only=False
            )
            ppnet = ppnet.to(self.device)
            ppnet.eval()
            wrapped = LogitsOnlyWrapper(ppnet).to(self.device)
            wrapped.eval()
            return wrapped

        except Exception as e:
            print(f"Failed to load v2 PPNet from {checkpoint_path}: {e}")
            raise
        finally:
            # Clean up added paths
            for p in paths_added:
                if p in sys.path:
                    sys.path.remove(p)
                    
    def _create_mambavision(
        self,
        checkpoint_path: Optional[str] = None,
        model_variant: str = "mamba_vision_L2",
        num_classes: int = 2,
    ) -> nn.Module:
        """Load MambaVision model with grayscale adaptation."""
        from mambavision import create_model as create_mamba_model

        # Create architecture
        model = create_mamba_model(model_variant, pretrained=False, num_classes=num_classes)

        # Adapt first conv for grayscale (3 → 1 channel)
        for name, module in model.named_modules():
            if isinstance(module, nn.Conv2d) and module.in_channels == 3:
                module.in_channels = 1
                module.weight = nn.Parameter(module.weight[:, :1, :, :].clone())
                break

        # Load checkpoint
        if checkpoint_path is not None:
            checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)

            if isinstance(checkpoint, dict):
                if "state_dict" in checkpoint:
                    state_dict = checkpoint["state_dict"]
                elif "model_state_dict" in checkpoint:
                    state_dict = checkpoint["model_state_dict"]
                elif "model" in checkpoint:
                    state_dict = checkpoint["model"]
                else:
                    state_dict = checkpoint
            else:
                state_dict = checkpoint

            # Remove 'module.' prefix if present
            state_dict = {
                (k[7:] if k.startswith("module.") else k): v
                for k, v in state_dict.items()
            }

            model.load_state_dict(state_dict, strict=False)

        model = model.to(self.device)
        model.eval()
        return model
</file>

<file path="utils.py">
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
import os

# ------------ Voters specific utils -----------------

def GetVoterValidation(batchSize):
    valData = torch.load("./data/kaleel_final_dataset_val_OnlyBubbles_Grayscale.pth", weights_only=False)
    valImages = valData["data"].float()
    valLabels = valData["binary_labels"].long()
    
    valDataset = TensorDataset(valImages, valLabels)
    valLoader = DataLoader(valDataset, batch_size=batchSize, shuffle=False)
    return valLoader

def GetVoterValidationCombined(batchSize):
    valData = torch.load("./data/kaleel_final_dataset_val_Combined_Grayscale.pth", weights_only=False)
    valImages = valData["data"].float()
    valLabels = valData["binary_labels"].long()
    
    valDataset = TensorDataset(valImages, valLabels)
    valLoader = DataLoader(valDataset, batch_size=batchSize, shuffle=False)
    return valLoader

def GetVoterTraining(batchSize):
    trainData = torch.load("./data/kaleel_final_dataset_train_OnlyBubbles_Grayscale.pth", weights_only=False)
    trainImages = trainData["data"].float()
    trainLabels = trainData["binary_labels"].long()
    
    trainDataset = TensorDataset(trainImages, trainLabels)
    trainLoader = DataLoader(trainDataset, batch_size=batchSize, shuffle=False)
    return trainLoader

def GetVoterTrainingCombined(batchSize):
    trainData = torch.load("./data/kaleel_final_dataset_train_Combined_Grayscale.pth", weights_only=False)
    trainImages = trainData["data"].float()
    trainLabels = trainData["binary_labels"].long()
    
    trainDataset = TensorDataset(trainImages, trainLabels)
    trainLoader = DataLoader(trainDataset, batch_size=batchSize, shuffle=True)
    return trainLoader


def GetVoterTrainingBalanced(batchSize, totalSamples, numClasses):
    # Get all training data (shuffled) with same batchSize
    fullTrainLoader = GetVoterTraining(batchSize=batchSize)
    
    # Collect all shuffled data from batches
    allImages = []
    allLabels = []
    for images, labels in fullTrainLoader:
        allImages.append(images)
        allLabels.append(labels)
    
    trainImages = torch.cat(allImages, dim=0)
    trainLabels = torch.cat(allLabels, dim=0)
    
    # Calculate samples per class
    samplesPerClass = totalSamples // numClasses
    
    # Get shape of images
    imgShape = trainImages[0].shape
    
    # Initialize tensors for balanced data
    balancedImages = torch.zeros(totalSamples, imgShape[0], imgShape[1], imgShape[2])
    balancedLabels = torch.zeros(totalSamples)
    
    # Track how many samples we've collected per class
    classCount = torch.zeros(numClasses)
    
    # Collect balanced samples
    currentIndex = 0
    for i in range(len(trainLabels)):
        label = int(trainLabels[i])
        
        if classCount[label] < samplesPerClass:
            balancedImages[currentIndex] = trainImages[i]
            balancedLabels[currentIndex] = label
            classCount[label] += 1
            currentIndex += 1
        
        if currentIndex >= totalSamples:
            break
    
    # Verify we got enough samples
    for c in range(numClasses):
        if classCount[c] != samplesPerClass:
            raise ValueError(f"Not enough samples for class {c}. Got {int(classCount[c])}, needed {samplesPerClass}")
    
    print(f"Balanced training data: {totalSamples} samples ({samplesPerClass} per class)")
    
    # Create dataloader
    balancedDataset = TensorDataset(balancedImages, balancedLabels.long())
    balancedLoader = DataLoader(balancedDataset, batch_size=batchSize, shuffle=False)
    
    return balancedLoader

# Calculate and print class-wise accuracy for a given model and dataloader
def calculateClasswiseAccuracy(dataLoader, model, device, numClasses):
    model.eval()
    
    # Initialize counters for each class
    correct_per_class = {i: 0 for i in range(numClasses)}
    total_per_class = {i: 0 for i in range(numClasses)}
    
    with torch.no_grad():
        for inputs, labels in dataLoader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            
            # Count correct predictions per class
            for label, pred in zip(labels, predicted):
                label_idx = label.item()
                total_per_class[label_idx] += 1
                if label_idx == pred.item():
                    correct_per_class[label_idx] += 1
    
    # Calculate accuracies
    classwise_acc = {}
    print(f"\n{'='*50}")
    print(f"Class-wise Accuracy")
    print(f"{'='*50}")
    print(f"{'Class':<10} {'Correct':<10} {'Total':<10} {'Accuracy':<10}")
    print(f"{'-'*50}")
    
    total_correct = 0
    total_samples = 0
    
    for cls in range(numClasses):
        if total_per_class[cls] > 0:
            acc = correct_per_class[cls] / total_per_class[cls]
        else:
            acc = 0.0
        classwise_acc[cls] = acc
        total_correct += correct_per_class[cls]
        total_samples += total_per_class[cls]
        
        print(f"{cls:<10} {correct_per_class[cls]:<10} {total_per_class[cls]:<10} {acc:.4f}")
    
    overall_acc = total_correct / total_samples if total_samples > 0 else 0.0
    print(f"{'-'*50}")
    print(f"{'Overall':<10} {total_correct:<10} {total_samples:<10} {overall_acc:.4f}")
    print(f"{'='*50}\n")
    
    return overall_acc, classwise_acc

# Scanned bubble data loader
def get_scanned_attack_loader(dataset_path, batch_size):
    """
    Load scanned bubble dataset for attack in UNet+Model mode.
    
    Args:
        dataset_path: Path to scanned bubble dataset
        batch_size: Batch size for dataloader
    
    Returns:
        DataLoader with scanned bubble samples
    """
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Scanned dataset not found: {dataset_path}")
    
    data = torch.load(dataset_path, weights_only=False)
    
    # Handle different data formats
    if "xData" in data:
        images = data["xData"].float()
        labels = data["yDataBinary"].long()
    elif "data" in data:
        images = data["data"].float()
        labels = data["binary_labels"].long()
    else:
        raise ValueError(f"Unknown data format. Keys: {list(data.keys())}")
    
    print(f"    Loaded {len(images)} scanned samples")
    print(f"    Shape: {images.shape}")
    print(f"    Labels: {torch.bincount(labels).tolist()}")
    
    dataset = TensorDataset(images, labels)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    
    return loader
# --------------------------------------------

#Convert a dataloader into x and y tensors 
def DataLoaderToTensor(dataLoader):
    #First check how many samples in the dataset
    numSamples = len(dataLoader.dataset) 
    sampleShape = GetOutputShape(dataLoader) #Get the output shape from the dataloader
    sampleIndex = 0
    #xData = torch.zeros(numSamples, sampleShape[0], sampleShape[1], sampleShape[2])
    xData = torch.zeros((numSamples,) + sampleShape) #Make it generic shape for non-image datasets
    yData = torch.zeros(numSamples)
    #Go through and process the data in batches 
    for i, (input, target) in enumerate(dataLoader):
        batchSize = input.shape[0] #Get the number of samples used in each batch
        #Save the samples from the batch in a separate tensor 
        for batchIndex in range(0, batchSize):
            xData[sampleIndex] = input[batchIndex]
            yData[sampleIndex] = target[batchIndex]
            sampleIndex = sampleIndex + 1 #increment the sample index 
    return xData, yData

#Convert a X and Y tensors into a dataloader
#Does not put any transforms with the data  
def TensorToDataLoader(xData, yData, transforms= None, batchSize=None, randomizer = None):
    if batchSize is None: #If no batch size put all the data through 
        batchSize = xData.shape[0]
    dataset = MyDataSet(xData, yData, transforms)
    if randomizer == None: #No randomizer
        dataLoader = torch.utils.data.DataLoader(dataset=dataset,  batch_size=batchSize, shuffle=False)
    else: #randomizer needed 
        train_sampler = torch.utils.data.RandomSampler(dataset)
        dataLoader = torch.utils.data.DataLoader(dataset=dataset,  batch_size=batchSize, sampler=train_sampler, shuffle=False)
    return dataLoader

def TensorToNumpy(x_tensor, y_tensor):
    x_numpy = x_tensor.cpu().numpy()
    y_numpy = y_tensor.cpu().numpy().astype(np.int64)
    return x_numpy, y_numpy

def NumpyToTensor(x_numpy, y_numpy):
    x_tensor = torch.from_numpy(x_numpy).float()
    y_tensor = torch.from_numpy(y_numpy).long()  # long is int64
    return x_tensor, y_tensor


# Find the actual min and max pixel values in the dataset
def GetDataBounds(dataLoader, device):
    minVal = float('inf')
    maxVal = float('-inf')
    
    for xData, _ in dataLoader:
        xData = xData.to(device)
        batchMin = xData.min().item()
        batchMax = xData.max().item()
        
        if batchMin < minVal:
            minVal = batchMin
        if batchMax > maxVal:
            maxVal = batchMax
    
    return minVal, maxVal

#Validate using a dataloader 
def validateD(valLoader, model, device=None):
    #switch to evaluate mode
    model.eval()
    acc = 0
    batchTracker = 0
    with torch.no_grad():
        #Go through and process the data in batches 
        for i, (input, target) in enumerate(valLoader):
            sampleSize = input.shape[0] #Get the number of samples used in each batch
            batchTracker = batchTracker + sampleSize
            #print("Processing up to sample=", batchTracker)
            if device == None: #assume cuda
                inputVar = input.cuda()
            else:
                inputVar = input.to(device)
            #compute output
            output = model(inputVar)
            output = output.float()
            #Go through and check how many samples correctly identified
            for j in range(0, sampleSize):
                if output[j].argmax(axis=0) == target[j]:
                    acc = acc +1
    acc = acc / float(len(valLoader.dataset))
    return acc

def GetCorrectlyIdentifiedSamplesBalanced(model, totalSamplesRequired, dataLoader, numClasses, device=None):
    model.eval()
    sampleShape = GetOutputShape(dataLoader)
    xData, yData = DataLoaderToTensor(dataLoader)
    #Basic error checking 
    if totalSamplesRequired % numClasses != 0:
        raise ValueError("The total number of samples in not evenly divisable by the number of classes.")
    #Get the number of samples needed for each class
    numSamplesPerClass = int(totalSamplesRequired/numClasses) 
    correctlyClassifiedSamples = torch.zeros((numClasses, numSamplesPerClass, sampleShape[0], sampleShape[1], sampleShape[2]))
    sanityCounter = torch.zeros((numClasses))
    #yPred = model.predict(xData)
    yPred = predictD(dataLoader, numClasses, model, device)
    a = 0
    for i in range(0, xData.shape[0]): #Go through every sample 
        a = a + 1
        predictedClass = yPred[i].argmax(axis=0)
        trueClass = yData[i]#.argmax(axis=0) 
        currentSavedCount = int(sanityCounter[int(trueClass)]) #Check how may samples we previously saved from this class
        #If the network predicts the sample correctly and we haven't saved enough samples from this class yet then save it
        if predictedClass == trueClass and currentSavedCount<numSamplesPerClass:
            correctlyClassifiedSamples[int(trueClass), currentSavedCount] = xData[i] #Save the sample 
            sanityCounter[int(trueClass)] = sanityCounter[int(trueClass)] + 1 #Add one to the count of saved samples for this class
    #Now we have gone through the entire network, make sure we have enough samples
    for c in range(0, numClasses):
        if sanityCounter[c] != numSamplesPerClass:
            raise ValueError("The network does not have enough correctly predicted samples for this class.")
    #Assume we have enough samples now, restore in a properly shaped array 
    xCorrect = torch.zeros((totalSamplesRequired, xData.shape[1], xData.shape[2], xData.shape[3]))
    yCorrect = torch.zeros((totalSamplesRequired))
    currentIndex = 0 #indexing for the final array
    for c in range(0, numClasses): #Go through each class
        for j in range(0, numSamplesPerClass): #For each sample in the class store it 
            xCorrect[currentIndex] = correctlyClassifiedSamples[c,j]
            yCorrect[currentIndex] = c
            #yCorrect[currentIndex, c] = 1.0
            currentIndex = currentIndex + 1 
    #return xCorrect, yCorrect
    cleanDataLoader = TensorToDataLoader(xCorrect, yCorrect, transforms = None, batchSize = dataLoader.batch_size, randomizer = None)
    return cleanDataLoader

#Get the output shape from the dataloader
def GetOutputShape(dataLoader):
    for i, (input, target) in enumerate(dataLoader):
        return input[0].shape

#Replicate TF's predict method behavior 
def predictD(dataLoader, numClasses, model, device=None):
    numSamples = len(dataLoader.dataset)
    yPred = torch.zeros(numSamples, numClasses)
    #switch to evaluate mode
    model.eval()
    indexer = 0
    batchTracker = 0
    with torch.no_grad():
        #Go through and process the data in batches 
        for i, (input, target) in enumerate(dataLoader):
            sampleSize = input.shape[0] #Get the number of samples used in each batch
            batchTracker = batchTracker + sampleSize
            #print("Processing up to sample=", batchTracker)
            if device == None:
                inputVar = input.cuda()
            else:
                inputVar = input.to(device)
            #compute output
            output = model(inputVar)
            output = output.float()
            for j in range(0, sampleSize):
                yPred[indexer] = output[j]
                indexer = indexer + 1 #update the indexer regardless 
    return yPred

#Class to help with converting between dataloader and pytorch tensor 
class MyDataSet(torch.utils.data.Dataset):
    def __init__(self, x_tensor, y_tensor, transforms=None):
        self.x = x_tensor
        self.y = y_tensor
        self.transforms = transforms

    def __getitem__(self, index):
        if self.transforms is None: #No transform so return the data directly
            return (self.x[index], self.y[index])
        else: #Transform so apply it to the data before returning 
            return (self.transforms(self.x[index]), self.y[index])

    def __len__(self):
        return len(self.x)
</file>

<file path="AttackWrappersAdaptiveBlackBox.py">
import torch
from torch.utils.data import TensorDataset, DataLoader

from attacks.linf_attack import FGSM
from DataLoaderGiant import DataLoaderGiant
import utils
import AttackRunner

from datetime import date
import os 
global queryCounter

def AdaptiveAttack(saveTag, device, oracle, syntheticModel, numClasses, training_config, numAttackSamples, attackLoader):
    
    # Unpack training config
    numIterations = training_config["numIterations"]
    epochsPerIteration = training_config["epochsPerIteration"]
    epsForAug = training_config["epsForAug"]
    learningRate = training_config["learningRate"]
    optimizerName = training_config["optimizerName"]
    dataLoaderForTraining = training_config["dataLoaderForTraining"]
    valLoader = training_config["valLoader"]
    clipMin = training_config["clipMin"]
    clipMax = training_config["clipMax"]
    
    #Create place to save all files
    today = date.today()
    dateString = today.strftime("%B"+"-"+"%d"+"-"+"%Y, ") #Get the year, month, day
    experimentDateAndName = dateString + saveTag #Name of experiment with data 
    saveDir = os.path.join(os.getcwd(), experimentDateAndName)
    if not os.path.isdir(saveDir): #If not there, make the directory 
        os.makedirs(saveDir)

    #Place to save the results 
    os.chdir(saveDir)
    resultsTextFile = open(experimentDateAndName+", Results.txt","a+")

    # Define Query Counter
    global queryCounter
    queryCounter = 0

    # -------------------------------- TRAINING AND EVALUATION ------------------------------------------
    
    # Train Synthetic Model 
    print("Phase 1: Train Synthetic Model")
    TrainSyntheticModel("./", device, oracle, syntheticModel, numIterations, epochsPerIteration, epsForAug, learningRate, optimizerName, dataLoaderForTraining, numClasses, clipMin, clipMax)
    torch.save(syntheticModel, f"./SyntheticModel_{saveTag}")

    # Training completed, switch to evaluation mode
    syntheticModel.eval()

    Synthetic_valAcc = utils.validateD(valLoader, syntheticModel, device)
    print("ValLoader Accuracy on Synthetic Model:", Synthetic_valAcc)  

    Oracle_valAcc = utils.validateD(valLoader, oracle, device)
    print("ValLoader Accuracy on Oracle Model:", Oracle_valAcc)  
    print("\n" + "-"*60)

    print("Queries used:", queryCounter)

    # Write validation accuracies to file
    resultsTextFile.write("="*70 + "\n")
    resultsTextFile.write("Validation accuracy after training\n")
    resultsTextFile.write("="*70 + "\n")
    resultsTextFile.write(f"ValLoader Accuracy on Synthetic Model: {Synthetic_valAcc:.4f}\n")
    resultsTextFile.write(f"ValLoader Accuracy on Oracle Model: {Oracle_valAcc:.4f}\n")
    resultsTextFile.write("="*70 + "\n\n")
    resultsTextFile.write(f"Query Used: {queryCounter}\n")
    resultsTextFile.write("="*70 + "\n\n")

    # ---------------------------------- ATTACK AND EVALUATION -------------------------------------------

    print("Phase 2: Adversarial Attacks")
    # Create correctLoader for attack
    correctLoader = utils.GetCorrectlyIdentifiedSamplesBalanced(oracle, numAttackSamples, attackLoader, numClasses)
    
    # Run all attacks
    all_results = AttackRunner.run_all_attacks(
        device=device,
        oracle=oracle,
        synthetic_model=syntheticModel,
        correct_loader=correctLoader,
        num_classes=numClasses,
        results_file=resultsTextFile
    )

    resultsTextFile.close()
    os.chdir("..")
    
    print("\n" + "="*70)
    print("EXPERIMENT COMPLETE!")
    print("="*70)
    
    return all_results


def TrainSyntheticModel(saveDir, device, oracle, syntheticModel, numIterations, epochsPerIteration, epsForAug, learningRate, optimizerName, trainDataLoader, numClasses, clipMin, clipMax):
    # First re-label the training data according to the oracle 
    trainDataLoader = LabelDataUsingOracle(oracle, trainDataLoader, device)
    # Setup the training parameters 
    criterion = torch.nn.CrossEntropyLoss()
    # Check what optimizer to use
    if optimizerName == "adam":
        optimizer = torch.optim.Adam(syntheticModel.parameters(), lr=learningRate)
    elif optimizerName == "sgd":
        optimizer = torch.optim.SGD(syntheticModel.parameters(), lr=learningRate, momentum=0.9, weight_decay=0)
    else:
        raise ValueError("Optimizer name not recognized.")
    # Setup the giant data loader
    homeDir = "./"
    giantDataLoader = DataLoaderGiant(homeDir, trainDataLoader.batch_size)
    giantDataLoader.AddLoader("OriginalLoader", trainDataLoader)
    # Do one round of training with the currently labeled training data 
    TrainingStep(device, syntheticModel, giantDataLoader, epochsPerIteration, criterion, optimizer)
    # Data augmentation and training steps 
    for i in range(0, numIterations):
        print("Running synthetic model training iteration =", i)
        # Create the synthetic data using FGSM and the synthetic model 
        numDataLoaders = giantDataLoader.GetNumberOfLoaders()  # Find out how many loaders we have to iterate over
        # Go through and generate adversarial examples for each dataloader
        print("=Step 0: Generating data loaders...")
        for j in range(0, numDataLoaders):
            print("--Generating data loader=", j)
            currentLoader = giantDataLoader.GetLoaderAtIndex(j)
            syntheticDataLoaderUnlabeled = FGSM.FGSMNativePytorch(device, currentLoader, syntheticModel, epsForAug, clipMin, clipMax, targeted=False)
            # Memory clean up 
            del currentLoader
            # Label the synthetic data using the oracle 
            syntheticDataLoader = LabelDataUsingOracle(oracle, syntheticDataLoaderUnlabeled, device)
            # Memory clean up
            del syntheticDataLoaderUnlabeled
            giantDataLoader.AddLoader("DataLoader,iteration=" + str(i) + "batch=" + str(j), syntheticDataLoader)          
        # Combine the new synthetic data loader and the original data loader
        print("=Step 1: Training the synthetic model...")
        # Train on the new data 
        TrainingStep(device, syntheticModel, giantDataLoader, epochsPerIteration, criterion, optimizer)

# Try to match Keras "fit" function as closely as possible 
def TrainingStep(device, model, giantDataLoader, numEpochs, criterion, optimizer):
    # Switch into training mode 
    model.train()
    numDataLoaders = giantDataLoader.GetNumberOfLoaders()  # Find out how many loaders we have to iterate over
    for e in range(0, numEpochs):
        print("--Epoch=", e)
        # Go through all dataloaders 
        for loaderIndex in range(0, numDataLoaders):
            print("----Training on data loader=", loaderIndex)
            dataLoader = giantDataLoader.GetLoaderAtIndex(loaderIndex)
            # Go through all the samples in the loader
            for i, (input, target) in enumerate(dataLoader):
                targetVar = target.to(device).long()
                inputVar = input.to(device)
                # Compute output
                output = model(inputVar)
                loss = criterion(output, targetVar)
                # Compute gradient and do SGD step
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
        del dataLoader
        del inputVar
        del targetVar
        torch.cuda.empty_cache()

def LabelDataUsingOracle(oracle, dataLoader, device):
    oracle.eval()
    numSamples = len(dataLoader.dataset)

    #Update the query Counter
    global queryCounter
    queryCounter = queryCounter + numSamples
    
    # Collect all data, original labels, and predictions
    all_inputs = []
    all_original_labels = []
    all_predictions = []
    
    with torch.no_grad():
        for inputs, original_labels in dataLoader:
            inputs = inputs.to(device)
            outputs = oracle(inputs)
            
            # Get hard labels (argmax of predictions)
            predictions = outputs.argmax(dim=1)
            
            all_inputs.append(inputs.cpu())
            all_original_labels.append(original_labels.cpu())
            all_predictions.append(predictions.cpu())
    
    # Concatenate all batches
    xData = torch.cat(all_inputs, dim=0)
    yOriginal = torch.cat(all_original_labels, dim=0)
    yLabels = torch.cat(all_predictions, dim=0)
    
    # ----------------- LABEL COMPARISON STATISTICS -----------------
    print("\n" + "-"*60)
    print("LABEL COMPARISON STATISTICS")
    
    # Overall statistics
    same_labels = (yOriginal == yLabels).sum().item()
    different_labels = (yOriginal != yLabels).sum().item()
    total_samples = len(yLabels)
    
    print(f"Total Samples: {total_samples}")
    print(f"Labels Unchanged: {same_labels} ({100*same_labels/total_samples:.2f}%)")
    print(f"Labels Changed:   {different_labels} ({100*different_labels/total_samples:.2f}%)")
    print("-"*60)
    
    # Create new DataLoader with oracle labels
    labeledDataset = TensorDataset(xData, yLabels)
    dataLoaderLabeled = DataLoader(
        labeledDataset, 
        batch_size=dataLoader.batch_size, 
        shuffle=False
    )
    
    return dataLoaderLabeled
</file>

<file path="main.py">
# AdaptiveAttack.py

"""
Adaptive Black Box Attack - Compatible with Model Only AND UNet+Model Mode

This script attacks oracle models using a synthetic VGG11 model for transfer.
Supports both:
  - Model Only: Uses original validation data for attack
  - UNet+Model: Uses scanned bubble data for attack (with UNet denoiser)
  
Training always uses voter training data.
"""

import torch
import random
import os
from torch.utils.data import DataLoader, TensorDataset

from ModelFactory import ModelFactory
import AttackWrappersAdaptiveBlackBox
import utils
from constants import (
    CHECKPOINTS,
    UNET_CHECKPOINT,
    EXPERIMENTS_ALL,
    EXPERIMENTS_UNET_ALL,
    EXPERIMENTS_SNN_RESNET20
)

# ------ Setting seed for reproducibility -------
SEED = 20

def reset_seed(seed=SEED):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# Set seed at module load
reset_seed()

# ------- SYTHETIC MODEL AND ORACLE CONFIGURATION ------------

# Synthetic model for transfer attack (VGG11 - will be trained, no checkpoint needed)
SYNTHETIC_MODEL = "vgg11"

# Choose experiment mode: "model_only" or "unet"
ATTACK_MODE = "model_only"  # for Model only attacks
# ATTACK_MODE = "unet"        # for UNet+Model attacks

# For Attack mode = model_only
EXPERIMENTS_MODEL_ONLY = EXPERIMENTS_ALL

# For Attack mode = unet (UNet+Model attacks)
# EXPERIMENTS_UNET_MODE = EXPERIMENTS_UNET_ALL

# -------- ATTACK FUNCTION -----------

def run_attack_on_oracle(model_name: str, config: dict, synthetic_model_name: str):
    """
    Run adaptive black box attack on a single oracle model.
    
    Args:
        model_name: Name of the oracle model
        config: Configuration dict with ckpt_path, dataset_path, use_unet, etc.
        synthetic_model_name: Name of synthetic model for transfer (e.g., "vgg11")
    """

    # ------ Seed is reset in each indvidual oracle model initiation for reproducibility -----
    reset_seed(SEED)
    
    # Parameters
    batchSize = 64
    numClasses = 2
    numTrainingSamples = 2000
    numAttackSamples = 500
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Extract config
    use_unet = config.get("use_unet", False)
    model_ckpt = config["ckpt_path"]
    dataset_path = config["dataset_path"]  # Scanned data path for UNet mode
    unet_ckpt = config.get("unet_ckpt", UNET_CHECKPOINT) if use_unet else None
    
    mode_str = "UNet+Model" if use_unet else "Model Only"
    
    print(f"Oracle model: {model_name} ({mode_str})", flush=True)
    print(f"Synthetic model: {synthetic_model_name} (will be trained)", flush=True)
    
    # Load models
    factory = ModelFactory(device=device)
    
    # Load synthetic model (VGG11 - untrained, no checkpoint)
    syntheticModel = factory.get_model(synthetic_model_name)
    
    # Load oracle model (with or without UNet wrapper)
    if use_unet:
        oracle = factory.get_unet_model_wrapper(
            model_name=model_name,
            model_checkpoint=model_ckpt,
            unet_checkpoint=unet_ckpt,
        )
    else:
        oracle = factory.get_model(model_name, model_ckpt)
    
    # Create save tag
    unet_suffix = "_unet" if use_unet else ""
    saveTag = f"Adaptive_Attack_Oracle={model_name}{unet_suffix}_Synthetic={synthetic_model_name}"
    
    #-------------- DATA LOADERS ---------------
    
    # Training & Validation Dataloaders
    trainLoader = utils.GetVoterTrainingBalanced(batchSize, numTrainingSamples, numClasses)
    valLoader = utils.GetVoterValidation(batchSize)
    
    # Attack dataloder
    if use_unet:
        # UNet+Model: Use scanned bubble data for attack
        attackLoader = utils.get_scanned_attack_loader(dataset_path, batchSize)
    else:
        # Model Only: Use valLoader data for attack
        attackLoader = utils.GetVoterValidation(batchSize) 
    
    # Training config
    training_config = {
        "batchSize": batchSize,
        "numIterations": 4,    
        "epochsPerIteration": 10,
        "epsForAug": 0.01,
        "learningRate": 0.0001,
        "numTrainingSamples": numTrainingSamples,
        "dataLoaderForTraining": trainLoader,
        "valLoader": valLoader,
        "optimizerName": "adam",
        "clipMin": 0.0,
        "clipMax": 1.0,
    }
    
    # Run attack
    try:
        AttackWrappersAdaptiveBlackBox.AdaptiveAttack(
            saveTag=saveTag,
            device=device,
            oracle=oracle,
            syntheticModel=syntheticModel,
            numClasses=numClasses,
            training_config=training_config,
            numAttackSamples=numAttackSamples,
            attackLoader=attackLoader,
        )
        print(f"Attack completed for {model_name}", flush=True)
        return True
    except Exception as e:
        print(f"Attack failed for {model_name}: {e}", flush=True)
        import traceback
        traceback.print_exc()
        return False

def main():
    # Select experiments based on mode
    if ATTACK_MODE == "unet":
        experiments = EXPERIMENTS_UNET_MODE
        mode_name = "UNet+Model"
        results_file = "adaptive_unet_results.txt"
    else:
        experiments = EXPERIMENTS_MODEL_ONLY
        mode_name = "Model Only"
        results_file = "adaptive_results.txt"
    
    results = {}
    
    for model_name, config in experiments.items():
        success = run_attack_on_oracle(
            model_name=model_name,
            config=config,
            synthetic_model_name=SYNTHETIC_MODEL,
        )
        results[model_name] = "Success" if success else "Failed"
    
    # Save results
    with open(results_file, "a", encoding="utf-8") as f:
        header = f" ADAPTIVE BLACK BOX ATTACK ({mode_name}) RESULTS "
        f.write("\n" + "=" * 10 + header + "=" * 10 + "\n")
        for model, status in results.items():
            use_unet = experiments[model].get("use_unet", False)
            unet_suffix = "_unet" if use_unet else ""
            f.write(f"{model}{unet_suffix:<40}: {status}\n")


if __name__ == '__main__':
    main()
</file>

</files>
