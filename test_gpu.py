import torch
import time

print("PyTorch version:", torch.__version__)
print("CUDA support:", torch.version.cuda)
print("GPU disponibile:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("Nome GPU:", torch.cuda.get_device_name(0))

def run_test(device, size=(4000, 4000)):
    print(f"\nEsecuzione test su {device} con tensori {size}...")
    
    # Creazione di due matrici casuali
    a = torch.randn(size, device=device)
    b = torch.randn(size, device=device)

    # Misura del tempo di moltiplicazione
    start = time.time()
    result = torch.matmul(a, b)
    # Se GPU, sincronizza per avere tempo reale
    if device.type == 'cuda':
        torch.cuda.synchronize()
    end = time.time()

    print(f"Tempo di esecuzione: {end - start:.6f} secondi")
    print(f"Valori di esempio della prima riga: {result[0][:5]}")

def main():
    # Verifica GPU
    if torch.cuda.is_available():
        gpu_device = torch.device("cuda")
        print(f"GPU rilevata: {torch.cuda.get_device_name(0)}")
        run_test(gpu_device)
    else:
        print("GPU non rilevata, verrà utilizzata la CPU.")

    # Esegui anche su CPU per confronto
    cpu_device = torch.device("cpu")
    run_test(cpu_device)

if __name__ == "__main__":
    main()