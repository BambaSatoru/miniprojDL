import sys
import pygame
import chess

def init_pygame():
    pygame.init()
    pygame.mixer.init()

WIDTH, HEIGHT = 600, 600
SQUARE_SIZE = WIDTH // 8
WHITE = (238, 238, 210)
BLACK = (118, 150, 86)
HIGHLIGHT = (186, 202, 68)

MOVE_SND = "assets/sounds/move.mp3"
CAPTURE_SND = "assets/sounds/capture.mp3"
PROMO_SND = "assets/sounds/promote.mp3"

def setup_ui():
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("DRL Chess")
    move_sound = pygame.mixer.Sound(MOVE_SND)
    capture_sound = pygame.mixer.Sound(CAPTURE_SND)
    promo_sound = pygame.mixer.Sound(PROMO_SND)
    return screen, move_sound, capture_sound, promo_sound

def draw_board(screen, board, selected=None, legal_moves=None):
    legal_moves = legal_moves or []
    for r in range(8):
        for c in range(8):
            rect = pygame.Rect(c*SQUARE_SIZE, r*SQUARE_SIZE, SQUARE_SIZE, SQUARE_SIZE)
            color = WHITE if (r + c) % 2 == 0 else BLACK
            pygame.draw.rect(screen, color, rect)
            sq = chess.square(c, 7 - r)
            if selected == sq:
                pygame.draw.rect(screen, HIGHLIGHT, rect, 4)
            if sq in legal_moves:
                pygame.draw.circle(screen, (255,255,0), rect.center, 8)
            piece = board.piece_at(sq)
            if piece:
                key = ('w' if piece.color else 'b') + '_' + {
                    chess.PAWN:'pawn', chess.ROOK:'rook', chess.KNIGHT:'knight',
                    chess.BISHOP:'bishop', chess.QUEEN:'queen', chess.KING:'king'
                }[piece.piece_type]
                img = pygame.transform.scale(
                    pygame.image.load(f"assets/pieces/{key}.png"),
                    (SQUARE_SIZE, SQUARE_SIZE)
                )
                screen.blit(img, rect.topleft)

def get_square(pos):
    x, y = pos
    col = x // SQUARE_SIZE
    row = y // SQUARE_SIZE
    return chess.square(col, 7 - row)

def play_sound(move, board, move_sound, capture_sound, promo_sound):
    if board.is_capture(move):
        capture_sound.play()
    elif move.promotion:
        promo_sound.play()
    else:
        move_sound.play()

def ask_promotion_choice(screen, board):
    font = pygame.font.Font(None, 40)
    clock = pygame.time.Clock()
    btns = []
    labels = [("Queen", chess.QUEEN), ("Rook", chess.ROOK), ("Bishop", chess.BISHOP), ("Knight", chess.KNIGHT)]
    for i, (txt, piece) in enumerate(labels):
        rect = pygame.Rect(WIDTH//2 - 170 + i*110, HEIGHT//2 - 25, 100, 50)
        btns.append((rect, txt, piece))
    while True:
        draw_board(screen, board)
        overlay = pygame.Surface((WIDTH, HEIGHT), pygame.SRCALPHA)
        overlay.fill((0,0,0,150))
        screen.blit(overlay, (0,0))
        for rect, txt, _ in btns:
            pygame.draw.rect(screen, (255,255,0), rect)
            screen.blit(font.render(txt, True, (0,0,0)), (rect.x+10, rect.y+10))
        pygame.display.flip()
        clock.tick(30)
        for ev in pygame.event.get():
            if ev.type == pygame.QUIT:
                pygame.quit(); sys.exit()
            if ev.type == pygame.MOUSEBUTTONDOWN:
                for rect, _, piece in btns:
                    if rect.collidepoint(ev.pos):
                        return piece