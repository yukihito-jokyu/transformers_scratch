import torch
import torch.optim as optim
from torch.utils.data import DataLoader

from src.transformers_scratch.Dataset import CostumDataset
from src.transformers_scratch.Transformer import Transformer
from transformers_scratch.Training import Training

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

src_vocab_size = 146829
tgt_vocab_size = 221860
max_len = 100
d_model = 12
head = 2
N = 2
batch_size = 12

src_vocab_path = "vocab/src_vocab.json"
tgt_vocab_path = "vocab/tgt_vocab.json"
src_dataset_path = "dataset/kyoto-train.ja"
tgt_dataset_path = "dataset/kyoto-train.en"


dataset = CostumDataset(
    src_vocab_path, tgt_vocab_path, src_dataset_path, tgt_dataset_path, max_seq=max_len
)

data_loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True)


net = Transformer(
    src_vocab_size=src_vocab_size,
    tgt_vocab_size=tgt_vocab_size,
    max_len=max_len,
    d_model=d_model,
    head=head,
    N=N,
    device=device,
)

optimizer = optim.Adam(net.parameters(), lr=0.001)

train = Training(net=net, optimizer=optimizer, device=device)

loss_list = train.fit(train_loader=data_loader)

print(loss_list)
