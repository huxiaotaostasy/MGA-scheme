import torch
from torch.nn import DataParallel
from torch.nn.parallel import DistributedDataParallel


def load_model(model, model_path, strict=True, cpu=False):
    if isinstance(model, DataParallel) or isinstance(model, DistributedDataParallel):
        model = model.module
    if cpu:
        loaded_model = torch.load(model_path, map_location='cpu')
    else:
        loaded_model = torch.load(model_path)
    model.load_state_dict(loaded_model, strict=strict)

def load_corresponding_model(model, state_dict, print_stats=True):
    num_matched, num_total = 0, 0
    curr_state_dict = model.state_dict()
    # print(curr_state_dict.keys())
    # print(state_dict.keys())
    for key in state_dict.keys():
        num_total += 1
        curr_key = key
        if curr_key in curr_state_dict and curr_state_dict[curr_key].shape == state_dict[key].shape:
            curr_state_dict[curr_key] = state_dict[key]
            num_matched += 1
    model.load_state_dict(curr_state_dict)
    if print_stats:
        print(f'Loaded state_dict: {num_matched}/{num_total} matched')

def load_base(model, state_dict, print_stats=True):
    """
    Only loads weights that matched in key and shape. Ignore other weights.
    """
    num_matched, num_total = 0, 0
    curr_state_dict = model.state_dict()
    # print(curr_state_dict.keys())
    # print(state_dict.keys())
    for key in state_dict.keys():
        num_total += 1
        curr_key = 'base.'+key
        if curr_key in curr_state_dict and curr_state_dict[curr_key].shape == state_dict[key].shape:
            curr_state_dict[curr_key] = state_dict[key]
            num_matched += 1
    model.load_state_dict(curr_state_dict)
    if print_stats:
        print(f'Loaded state_dict: {num_matched}/{num_total} matched')

def load_base_T(model, state_dict, print_stats=True):
    """
    Only loads weights that matched in key and shape. Ignore other weights.
    """
    num_matched, num_total = 0, 0
    curr_state_dict = model.state_dict()
    # print(curr_state_dict.keys())
    # print(state_dict.keys())
    for key in curr_state_dict.keys():
        num_total += 1
        curr_key = 'base.'+key
        if curr_key in state_dict and curr_state_dict[key].shape == state_dict[curr_key].shape:
            curr_state_dict[key] = state_dict[curr_key]
            num_matched += 1
    model.load_state_dict(curr_state_dict)
    if print_stats:
        print(f'Loaded state_dict: {num_matched}/{num_total} matched')

def load_backbone(model, state_dict, print_stats=True):
    """
    Only loads weights that matched in key and shape. Ignore other weights.
    """
    num_matched, num_total = 0, 0
    curr_state_dict = model.state_dict()
    for key in state_dict.keys():
        num_total += 1
        curr_key = 'backbone.'+key
        if curr_key in curr_state_dict and curr_state_dict[curr_key].shape == state_dict[key].shape:
            curr_state_dict[curr_key] = state_dict[key]
            num_matched += 1
    model.load_state_dict(curr_state_dict)
    if print_stats:
        print(f'Loaded state_dict: {num_matched}/{num_total} matched')


def load_solver(optimizer, lr_scheduler, solver_path):
    loaded_solver = torch.load(solver_path)
    loaded_optimizer = loaded_solver['optimizer']
    loaded_lr_scheduler = loaded_solver['lr_scheduler']
    iteration = loaded_solver['iteration']
    optimizer.load_state_dict(loaded_optimizer)
    lr_scheduler.load_state_dict(loaded_lr_scheduler)

    return iteration


def save_model(model, model_path):
    if isinstance(model, DataParallel) or isinstance(model, DistributedDataParallel):
        model = model.module
    torch.save(model.state_dict(), model_path)


def save_solver(optimizer, lr_scheduler, iteration, solver_path):
    solver = dict()
    solver['optimizer'] = optimizer.state_dict()
    solver['lr_scheduler'] = lr_scheduler.state_dict()
    solver['iteration'] = iteration
    torch.save(solver, solver_path)
