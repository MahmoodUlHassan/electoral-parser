from parser.models import OcrToken, PageMeta


def tok(text: str, x: float = 0, y: float = 0, conf: float = 0.99) -> OcrToken:
    return OcrToken(
        text=text,
        confidence=conf,
        bbox=[[x, y], [x + 40, y], [x + 40, y + 12], [x, y + 12]],
    )


CARD_TOKENS = [
    tok("1", 8, 8),
    tok("SWD7756588", 220, 8),
    tok("Name : Pavankumar Illa", 10, 50),
    tok("Fathers Name: Srinivas Illa", 10, 78),
    tok("House Number : 001", 10, 106),
    tok("Age : 25 Gender : Male", 10, 134),
]
