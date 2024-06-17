import os
import platform

if not os.path.isfile("ocr_data.zip"):
    if not os.path.isdir("data"):
        print("upload images")
        exit()
elif platform.system() == "Linux":
    os.system(
        "git clone https://github.com/n1teshy/seq-transduction && mv seq-transduction/core . && rm -rf seq-transduction"
    )
    os.system(
        "git clone https://github.com/n1teshy/cache && mv cache/ocr/tokenizers . && rm -rf cache"
    )
    os.system("unzip ocr_data.zip -d . > /dev/null && rm ocr_data.zip")
else:
    os.system(
        "git clone https://github.com/n1teshy/seq-transduction & move seq-transduction/core . & rd /s /q seq-transduction"
    )
    os.system(
        "git clone https://github.com/n1teshy/cache & move cache/ocr/tokenizers . & rd /s /q cache"
    )
    os.system(
        "powershell Expand-Archive -Path ocr_data.zip -DestinationPath . > NUL & del ocr_data.zip"
    )


import torch
import signal
import threading
from collections import deque
from core.config import device
import torch.nn.functional as F
from core.models import OCR, resnet14 as cnn_fn
from torch.utils.data import DataLoader
from core.datasets.image import OCRDataset
from core.tokenizers.regex import get_tokenizer
from core.utils import get_param_count, DualLogger, kaiming_init

logger = DualLogger("model.log")
interruption = threading.Event()
signal.signal(signal.SIGINT, lambda _, __: interruption.set())


EPOCHS = 100
LEARNING_RATE = 0.003
EMBEDDING_SIZE = 256
VOCAB_SZE = None
MAX_LEN = 100
DEC_LAYERS = 5
DEC_HEADS = 8
PADDING_ID = None
MIN_PROGRESS = 0.05
MAX_LOSS_DIFF = 0.25
BATCH_SIZE = 4
TRAIN_FOLDER = "data/train"
VAL_FOLDER = "data/test"
ENCODER = cnn_fn(num_classes=EMBEDDING_SIZE)

mean_window = 400
accumulation_steps = 4
cur_loss_wt = 1 / mean_window
mn_loss_wt = 1 - cur_loss_wt
param_dir = f"cnn_{cnn_fn.__name__}_emb_{EMBEDDING_SIZE}_lyrs_{DEC_LAYERS}_hds_{DEC_HEADS}_mxlen_{MAX_LEN}"
os.makedirs(param_dir, exist_ok=True)

last_saved_at = float("inf")
t_loss_sum, v_loss_sum = 0, 0
t_losses, v_losses = deque(maxlen=mean_window), deque(maxlen=mean_window)


tokenizer = get_tokenizer("", 256, "tokenizers/en")
VOCAB_SZE = tokenizer.size

train_dataset = OCRDataset(
    TRAIN_FOLDER, mapping_file="meta/labels.txt", tokenizer=tokenizer
)
val_dataset = OCRDataset(
    VAL_FOLDER, mapping_file="meta/labels.txt", tokenizer=tokenizer
)
PADDING_ID = train_dataset.pad_id

train_dataloader = DataLoader(
    train_dataset, collate_fn=train_dataset.collate, batch_size=BATCH_SIZE, shuffle=True
)
val_dataloader = DataLoader(
    val_dataset, collate_fn=val_dataset.collate, batch_size=BATCH_SIZE, shuffle=True
)
logger.log(f"train dataloader: {len(train_dataloader)} batches")
logger.log(f"val dataloader: {len(val_dataloader)} batches")


model = OCR.spawn(
    encoder=ENCODER.to(device),
    out_vocab_size=VOCAB_SZE,
    embedding_size=EMBEDDING_SIZE,
    max_len=MAX_LEN,
    dec_layers=DEC_LAYERS,
    dec_heads=DEC_HEADS,
    tgt_pad_id=PADDING_ID,
)
# model.load_state_dict(torch.load("ocr_params.pth", map_location=device))
kaiming_init(model)
model_param_count = get_param_count(model)
resnet_param_count = get_param_count(ENCODER)
logger.log(
    "parameters: %.4fmn (%.4f + %.4f)"
    % (
        model_param_count / 1e6,
        resnet_param_count / 1e6,
        (model_param_count - resnet_param_count) / 1e6,
    )
)
optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)


def update_loss_record(split, loss):
    global t_loss_sum, v_loss_sum
    losses = t_losses if split == "train" else v_losses
    first_val = 0
    if len(losses) == losses.maxlen:
        first_val = losses.popleft()
    if split == "train":
        t_loss_sum += loss - first_val
    else:
        v_loss_sum += loss - first_val
    losses.append(loss)
    return (t_loss_sum if split == "train" else v_loss_sum) / len(losses)


@torch.no_grad()
def get_loss(split, batches=accumulation_steps):
    model.eval()
    iters, acc_loss = 0, 0
    dataloader = train_dataloader if split == "train" else val_dataloader
    for pixels, tokens in dataloader:
        logits = model(pixels, tokens[:, :-1])
        B, T, C = logits.shape
        logits, tokens = logits.reshape((B * T, C)), tokens[:, 1:].reshape(-1)
        loss = F.cross_entropy(logits, tokens)
        acc_loss += loss.item()
        if iters == batches:
            break
        iters += 1
    model.train()
    return acc_loss / batches


def get_losses():
    return get_loss("train"), get_loss("val")


def save_model(t_loss, v_loss, folder=param_dir, checkpoint=False):
    filename = "ocr_%.4f_%.4f_class_%d_lr_%.4f.%s" % (
        t_loss,
        v_loss,
        EMBEDDING_SIZE,
        LEARNING_RATE,
        "chk" if checkpoint else "pth",
    )
    torch.save(model.state_dict(), os.path.join(folder, f"{filename}"))
    logger.log("saved model with losses %.4f/%.4f at %s" % (t_loss, v_loss, filename))


def progress_monitor(e_no, b_no, t_loss, v_loss, mt_loss, mv_loss):
    global last_saved_at
    logger.log(
        "%d:%d -> %.4f(mean:%.4f), %.4f(mean:%.4f)"
        % (e_no, b_no, t_loss, mt_loss, v_loss, mv_loss)
    )
    if (
        e_no > 1 or len(t_losses) >= mean_window
    ) and last_saved_at - mv_loss >= MIN_PROGRESS:
        if abs(mt_loss - mv_loss) < MAX_LOSS_DIFF:
            save_model(mt_loss, mv_loss)
            last_saved_at = mv_loss
        else:
            word = "overfitting" if mt_loss - mv_loss > 0 else "underfitting"
            logger.log(f"the model seems to be {word}")
    if interruption.is_set():
        if input("save checkpoint? ").strip().startswith("y"):
            save_model(mt_loss, mv_loss, checkpoint=True)
        exit()


for e_no in range(1, EPOCHS + 1):
    for b_no, (pixels, tokens) in enumerate(train_dataloader, start=1):
        try:
            logits = model(pixels, tokens[:, :-1])
            B, T, C = logits.shape
            logits, tokens = logits.reshape((B * T, C)), tokens[:, 1:].reshape(-1)
            t_loss = F.cross_entropy(logits, tokens)
            (t_loss / accumulation_steps).backward()
            if b_no % accumulation_steps == 0 or b_no == len(train_dataloader):
                optimizer.step()
                optimizer.zero_grad()
                t_loss, v_loss = get_losses()
                mt_loss = update_loss_record("train", t_loss)
                mv_loss = update_loss_record("val", v_loss)
                progress_monitor(e_no, b_no, t_loss, v_loss, mt_loss, mv_loss)
        except Exception:
            save_model(mt_loss, mv_loss, checkpoint=True)
            raise

# model.eval()
# def predict(image, bos_id, eos_id):
#     context = torch.tensor([[bos_id]], device=device)
#     while True:
#         logits = model(image, context)
#         probs = F.softmax(logits, dim=-1)
#         probs = probs.view(-1, probs.shape[-1])
#         choices = torch.multinomial(probs, num_samples=1)
#         choices = choices[-1, :]
#         if choices.item() == eos_id:
#             break
#         context = torch.cat((context, choices.unsqueeze(0)), dim=1)
#     return context[0].tolist()


# for images, tgt_tokens in test_dataloader:
#     image = images[0].unsqueeze(0)
#     pred_tokens = predict(image, val_dataset.bos_id, val_dataset.eos_id)
#     print(tokenizer.decode(tgt_tokens[0].tolist()), tokenizer.decode(pred_tokens))
