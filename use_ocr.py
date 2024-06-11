import os
import platform

if not os.path.isfile("ocr_data.zip"):
    if not os.path.isdir("data"):
        print("upload images")
        exit()
elif platform.system() == "Linux":
    os.system(
        "git clone https://github.com/n1teshy/sequence-transduction && mv sequence-transduction/core . && rm -rf sequence-transduction"
    )
    os.system(
        "git clone https://github.com/n1teshy/cache && mv cache/ocr/tokenizers . && rm -rf cache"
    )
    os.system("unzip ocr_data.zip -d . > /dev/null && rm ocr_data.zip")
else:
    os.system(
        "git clone https://github.com/n1teshy/sequence-transduction & move sequence-transduction/core . & rd /s /q sequence-transduction"
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
from core.config import device
import torch.nn.functional as F
from core.models import OCR, resnet18
from torch.utils.data import DataLoader
from core.datasets.image import OCRDataset
from core.tokenizers.regex import get_tokenizer
from core.utils import get_param_count, DualLogger, kaiming_init

cnn_fn = resnet18
logger = DualLogger("model.log")
interrupted = threading.Event()
signal.signal(signal.SIGINT, lambda _, __: interrupted.set())


EPOCHS = 100
LEARNING_RATE = 0.003
EMBEDDING_SIZE = 288
VOCAB_SZE = None
MAX_LEN = 100
DEC_LAYERS = 5
DEC_HEADS = 8
PADDING_ID = None
MIN_PROGRESS = 0.1
BATCH_SIZE = 8
TRAIN_FOLDER = "data/train"
VAL_FOLDER = "data/test"
ENCODER = cnn_fn(num_classes=EMBEDDING_SIZE)

mt_loss, mv_loss = None, None
last_saved_at = float("inf")
cur_loss_wt = 1 / 400
mn_loss_wt = 1 - cur_loss_wt
param_dir = f"cnn_{cnn_fn.__name__}_emb_{EMBEDDING_SIZE}_lyrs_{DEC_LAYERS}_hds_{DEC_HEADS}_mxlen_{MAX_LEN}"
os.makedirs(param_dir, exist_ok=True)


tokenizer = get_tokenizer("_.txt", 384, "tokenizers/en")
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
test_dataloader = DataLoader(
    val_dataset, collate_fn=val_dataset.collate, batch_size=BATCH_SIZE, shuffle=True
)


@torch.no_grad()
def get_test_loss(model):
    model.eval()
    for pixels, tokens in test_dataloader:
        break
    logits = model(pixels, tokens[:, :-1])
    B, T, C = logits.shape
    logits, tokens = logits.reshape((B * T, C)), tokens[:, 1:].reshape(-1)
    model.train()
    return F.cross_entropy(logits, tokens)


def save_model(folder=param_dir, checkpoint=False):
    filename = "ocr_%.4f_%.4f_class_%d_lr_%.4f.%s" % (
        mt_loss,
        mv_loss,
        EMBEDDING_SIZE,
        LEARNING_RATE,
        "chk" if checkpoint else "pth",
    )
    torch.save(model.state_dict(), os.path.join(folder, f"{filename}"))
    logger.log("saved model with losses %.4f/%.4f at %s" % (mt_loss, mv_loss, filename))


model = OCR.spawn(
    encoder=ENCODER.to(device),
    out_vocab_size=VOCAB_SZE,
    embedding_size=EMBEDDING_SIZE,
    max_len=MAX_LEN,
    dec_layers=DEC_LAYERS,
    dec_heads=DEC_HEADS,
    tgt_pad_id=PADDING_ID,
)
# model.load_state_dict(torch.load("", map_location=device))
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


for epoch in range(1, EPOCHS + 1):
    for batch, (pixels, tokens) in enumerate(train_dataloader, start=1):
        try:
            logits = model(pixels, tokens[:, :-1])
            B, T, C = logits.shape
            logits, tokens = logits.reshape((B * T, C)), tokens[:, 1:].reshape(-1)
            t_loss = F.cross_entropy(logits, tokens)
            optimizer.zero_grad()
            t_loss.backward()
            optimizer.step()
            t_loss = t_loss.item()
            v_loss = get_test_loss(model).item()
            mt_loss = (mt_loss or t_loss) * mn_loss_wt + t_loss * cur_loss_wt
            mv_loss = (mv_loss or v_loss) * mn_loss_wt + v_loss * cur_loss_wt
            logger.log(
                "%d:%d -> %.4f(mean:%.4f), %.4f(mean:%.4f)"
                % (epoch, batch, t_loss, mt_loss, v_loss, mv_loss)
            )
            if last_saved_at - mv_loss >= MIN_PROGRESS:
                save_model()
                last_saved_at = mv_loss
            if interrupted.is_set():
                save_model(checkpoint=True)
                exit()
        except Exception:
            save_model(checkpoint=True)
            raise


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
