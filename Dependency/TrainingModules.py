from torch import nn
import torch
import numpy as np

def testing_epoch(network, testing_set, device, loss, classification_or_reconstruction=None):
    key = classification_or_reconstruction.strip().lower() if isinstance(classification_or_reconstruction, str) else None
    assert key in ("reconstruction", "classification"), \
        "classification_or_reconstruction must be 'reconstruction' or 'classification'"
    is_classification = (key == "classification")

    network.eval()
    test_cost = []
    if is_classification:
        test_acc = []

    with torch.no_grad():
        for i, (batch_data, batch_labels) in enumerate(testing_set):
            batch_data = batch_data.to(device)
            batch_labels = batch_labels.to(device)

            pred = network(batch_data)

            if is_classification:
                if isinstance(loss, nn.CrossEntropyLoss):
                    error = loss(pred, batch_labels)
                elif isinstance(loss, nn.MSELoss):
                    target = torch.nn.functional.one_hot(
                        batch_labels, num_classes=pred.shape[-1]
                    ).to(pred.dtype)
                    error = loss(pred, target)
                else:
                    raise TypeError(
                        "For classification, loss must be nn.CrossEntropyLoss() or nn.MSELoss()"
                    )

                test_cost.append(float(error.detach().cpu().numpy()))
                test_acc.append((torch.argmax(pred, dim=1) == batch_labels).float().mean().item())
            else:
                error = loss(pred, batch_data)
                test_cost.append(float(error.detach().cpu().numpy()))

    test_cost = np.mean(test_cost)
    if is_classification:
        test_acc = np.mean(test_acc) * 100
        return network, test_cost, test_acc
    else:
        return network, test_cost

def training_epoch(network, training_set, device, optimiser, loss, current_epoch,
                   classification_or_reconstruction=None, lambda_psi=0.0):
    key = classification_or_reconstruction.strip().lower() if isinstance(classification_or_reconstruction, str) else None
    assert key in ("reconstruction", "classification"), \
        "classification_or_reconstruction must be 'reconstruction' or 'classification'"
    is_classification = (key == "classification")

    network.train()
    train_x = []
    train_cost = []
    if is_classification:
        train_acc = []

    for i, (batch_data, batch_labels) in enumerate(training_set):
        fractional_epoch = current_epoch + i / len(training_set)

        optimiser.zero_grad(set_to_none=True)

        batch_data = batch_data.to(device)
        batch_labels = batch_labels.to(device)

        pred = network(batch_data)

        if is_classification:
            if isinstance(loss, nn.CrossEntropyLoss):
                error = loss(pred, batch_labels)
            elif isinstance(loss, nn.MSELoss):
                target = torch.nn.functional.one_hot(
                    batch_labels, num_classes=pred.shape[-1]
                ).to(pred.dtype)
                error = loss(pred, target)
            else:
                raise TypeError(
                    "For classification, loss must be nn.CrossEntropyLoss() or nn.MSELoss()"
                )
        else:
            error = loss(pred, batch_data)

        if getattr(network, "linear_correction_approach", "").upper() == "TRAINABLE+DECAY":
            psi_decay = 0.0
            for psi in network.psi_parameters:
                psi_decay = psi_decay + psi.square().sum()
            error = error + lambda_psi * psi_decay

        train_x.append(fractional_epoch)
        train_cost.append(float(error.detach().cpu().numpy()))
        if is_classification:
            train_acc.append(100.0 * (torch.argmax(pred, dim=1) == batch_labels).float().mean().item())

        error.backward()
        optimiser.step()

    print("Done!")
    if is_classification:
        return network, train_x, train_cost, train_acc
    else:
        return network, train_x, train_cost

def training_loop(network, training_set, testing_set, epochs, learning_rate, device, quiet,
                      classification_or_reconstruction=None, lambda_psi=0.0, loss=nn.MSELoss(), optimiser=None):
    """
    Performs mean squared error training using an accuracy measure at every epoch.
    """
    key = classification_or_reconstruction.strip().lower() if isinstance(classification_or_reconstruction, str) else None
    assert key in ("reconstruction", "classification"), "classification_or_reconstruction must be 'reconstruction' or 'classification'"
    is_classification = (key == "classification")

    network.to(device)

    #
    if optimiser is None: optimiser = torch.optim.Adam(network.parameters(), lr=learning_rate)
    # loss = nn.MSELoss()
    loss = nn.CrossEntropyLoss()
    train_x = []
    test_x = []
    train_cost = []
    test_cost = []

    if is_classification:
        train_acc = []
        test_acc = []
        network, temp_cost, temp_acc = testing_epoch(network, testing_set, device, loss,
                                                     classification_or_reconstruction)
        test_acc.append(temp_acc)
    else:
        network, temp_cost = testing_epoch(network, testing_set, device, loss, classification_or_reconstruction)

    test_x.append(0)
    test_cost.append(temp_cost)

    if not quiet:
        print(f"Initial Testing Cost: {test_cost[-1]:5.4f} ", end="")
    if not quiet and is_classification:
        print(f"Accuracy={test_acc[-1]:5.4f}% ", end="")
    if not quiet:
        print("Beginning Network Training")

    for epoch in range(epochs):
        if not quiet:
            print(f"Starting Epoch {epoch + 1}/{epochs}... ", end="")

        if is_classification:
            network, epoch_x, epoch_cost, epoch_acc = training_epoch(
                network, training_set, device, optimiser, loss, epoch,
                classification_or_reconstruction, lambda_psi=lambda_psi
            )
            train_acc += epoch_acc
        else:
            network, epoch_x, epoch_cost = training_epoch(
                network, training_set, device, optimiser, loss, epoch,
                classification_or_reconstruction, lambda_psi=lambda_psi
            )

        train_x += epoch_x
        train_cost += epoch_cost

        if is_classification:
            network, temp_cost, temp_acc = testing_epoch(network, testing_set, device, loss,
                                                         classification_or_reconstruction)
            test_acc.append(temp_acc)
        else:
            network, temp_cost = testing_epoch(network, testing_set, device, loss, classification_or_reconstruction)

        test_x.append(epoch + 1)
        test_cost.append(temp_cost)

        if not quiet:
            print(f"Testing Cost: {test_cost[-1]:5.4f} ", end="")
        if not quiet and is_classification:
            print(f"Accuracy: {test_acc[-1]:5.4f}% ", end="")
        print()

    if not quiet:
        print("Totally Done!")

    if is_classification:
        return network.cpu(), [np.array(i) for i in [train_x, test_x, train_cost, test_cost, train_acc, test_acc]]
    else:
        return network.cpu(), [np.array(i) for i in [train_x, test_x, train_cost, test_cost]]