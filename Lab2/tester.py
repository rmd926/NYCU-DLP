import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from Dataloader import MIBCI2aDataset
from model.SCCNet import SCCNet

def test(model, device, test_loader, criterion):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += criterion(output, target).item()
            _, predicted = output.max(1)
            correct += predicted.eq(target).sum().item()
    
    test_loss /= len(test_loader)
    accuracy = 100. * correct / len(test_loader.dataset)
    print(f'Test set: Average loss: {test_loss:.4f}, Accuracy: {correct}/{len(test_loader.dataset)} ({accuracy:.2f}%)')
    return test_loss, accuracy

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    criterion = nn.CrossEntropyLoss()

    print("Select the model to test:")
    print("1. Subject Dependent (SD)")
    print("2. Leave-One-Subject-Out (LOSO)")
    print("3. Fine-Tuning (FT)")
    choice = input("Enter your choice (1, 2, or 3): ")

    if choice == '1':
        sd_test_dataset = MIBCI2aDataset(mode='test')
        sd_test_loader = DataLoader(sd_test_dataset, batch_size=64, shuffle=False)
        
        model_sd = SCCNet(num_classes=4, input_channels=22, Nu=22, Nt=1, dropout_rate=0.5, weight_decay = 1e-4)
        model_sd.load_state_dict(torch.load('./best_sccnet_model_SD_(61.46).pth', map_location=device))
        model_sd = model_sd.to(device)
        
        print("Testing Subject Dependent Model")
        test(model_sd, device, sd_test_loader, criterion)

    elif choice == '2':
        loso_test_dataset = MIBCI2aDataset(mode='test')
        loso_test_loader = DataLoader(loso_test_dataset, batch_size=64, shuffle=False)
        
        model_loso = SCCNet(num_classes=4, input_channels=22, Nu=22, Nt=1, dropout_rate=0.5,weight_decay = 1e-4)
        model_loso.load_state_dict(torch.load('./best_sccnet_model_LOSO(60.42).pth', map_location=device))
        model_loso = model_loso.to(device)
        
        print("Testing LOSO Model")
        test(model_loso, device, loso_test_loader, criterion)

    elif choice == '3':
        ft_test_dataset = MIBCI2aDataset(mode='test')
        ft_test_loader = DataLoader(ft_test_dataset, batch_size=64, shuffle=False)
        
        model_ft = SCCNet(num_classes=4, input_channels=22, Nu=22, Nt=1, dropout_rate=0.5,weight_decay = 1e-4)
        model_ft.load_state_dict(torch.load('./best_sccnet_model_LOSO_FT(73.26).pth', map_location=device))
        model_ft = model_ft.to(device)
        
        print("Testing Finetune Model")
        test(model_ft, device, ft_test_loader, criterion)

    else:
        print("Invalid choice. Please run the program again and select a valid option.")

if __name__ == '__main__':
    main()


# import torch
# import torch.nn as nn
# from torch.utils.data import DataLoader
# from Dataloader import MIBCI2aDataset
# from model.SCCNet import SCCNet

# def test(model, device, test_loader, criterion):
#     model.eval()
#     test_loss = 0
#     correct = 0
#     with torch.no_grad():
#         for data, target in test_loader:
#             data, target = data.to(device), target.to(device)
#             output = model(data)
#             test_loss += criterion(output, target).item()
#             _, predicted = output.max(1)
#             correct += predicted.eq(target).sum().item()
    
#     test_loss /= len(test_loader)
#     accuracy = 100. * correct / len(test_loader.dataset)
#     print(f'Test set: Average loss: {test_loss:.4f}, Accuracy: {correct}/{len(test_loader.dataset)} ({accuracy:.2f}%)')
#     return test_loss, accuracy

# def main():
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     criterion = nn.CrossEntropyLoss()

#     # Subject dependent testing
#     # sd_test_dataset = MIBCI2aDataset(mode='test')
#     # sd_test_loader = DataLoader(sd_test_dataset, batch_size=64, shuffle=False)
    
#     # model_sd = SCCNet(num_classes=4, input_channels=22, Nu=22, Nt=1, dropout_rate=0.3)
#     # model_sd.load_state_dict(torch.load('./best_sccnet_model_SD.pth', map_location=device))
#     # model_sd = model_sd.to(device)
    
#     # print("Testing Subject Dependent Model")
#     # test(model_sd, device, sd_test_loader, criterion)

#     # LOSO testing
    
#     loso_test_dataset = MIBCI2aDataset(mode='test')
#     loso_test_loader = DataLoader(loso_test_dataset, batch_size=64, shuffle=False)
    
#     model_loso = SCCNet(num_classes=4, input_channels=22, Nu=22, Nt=1, dropout_rate=0.3)
#     model_loso.load_state_dict(torch.load('./best_sccnet_model_SD(61.15new).pth', map_location=device))
#     model_loso = model_loso.to(device)
    
#     print("Testing LOSO Model")
#     test(model_loso, device, loso_test_loader, criterion)
#     '''
#     # Finetune testing
#     ft_test_dataset = MIBCI2aDataset(mode='test')
#     ft_test_loader = DataLoader(ft_test_dataset, batch_size=32, shuffle=False)
    
#     model_ft = SCCNet(num_classes=4, input_channels=22, Nu=22, Nt=1, dropout_rate=0.1)
#     model_ft.load_state_dict(torch.load('./best_sccnet_model_LOSO_FT.pth', map_location=device))
#     model_ft = model_ft.to(device)
    
#     print("Testing Finetune Model")
#     test(model_ft, device, ft_test_loader, criterion)
#     '''
# if __name__ == '__main__':
#     main()


