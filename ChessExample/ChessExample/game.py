import sys
import socket
import threading
import queue
import pygame
import chess
import torch
from dqn import load_dqn, board_to_tensor, move_to_idx
from server import host_game, join_game, BUFFER_SIZE
import interface
from interface import WIDTH, HEIGHT

interface.init_pygame()
screen, move_sound, capture_sound, promo_sound = interface.setup_ui()

def draw_overlay(message: str, color=(255,255,255)):
    overlay = pygame.Surface((WIDTH, HEIGHT), pygame.SRCALPHA)
    overlay.fill((0, 0, 0, 180))
    font = pygame.font.Font(None, 50)
    text = font.render(message, True, color)
    screen.blit(overlay, (0, 0))
    screen.blit(text, ((WIDTH - text.get_width()) // 2, (HEIGHT - text.get_height()) // 2))
    pygame.display.flip()

def local_play():
    board = chess.Board()
    policy_net = load_dqn()
    selected = None
    legal_moves = []
    turn = chess.WHITE
    clock = pygame.time.Clock()
    running = True

    while running:
        for ev in pygame.event.get():
            if ev.type == pygame.QUIT:
                pygame.quit(); sys.exit()
            if ev.type == pygame.MOUSEBUTTONDOWN and turn == chess.WHITE:
                sq = interface.get_square(ev.pos)
                if selected is not None and sq == selected:
                    selected, legal_moves = None, []
                elif board.piece_at(sq) and board.color_at(sq) == chess.WHITE:
                    selected = sq
                    legal_moves = [m.to_square for m in board.legal_moves if m.from_square == sq]
                elif selected is not None:
                    move = chess.Move(selected, sq)
                    p = board.piece_at(selected)
                    if p and p.piece_type == chess.PAWN and chess.square_rank(sq) == 7:
                        promo = interface.ask_promotion_choice(screen, board)
                        move = chess.Move(selected, sq, promotion=promo)
                    if move in board.legal_moves:
                        board.push(move)
                        interface.play_sound(move, board, move_sound, capture_sound, promo_sound)
                        selected, legal_moves = None, []
                        turn = chess.BLACK
                    else:
                        selected, legal_moves = None, []

        if turn == chess.BLACK and not board.is_game_over():
            state = board_to_tensor(board)
            with torch.no_grad():
                qvals = policy_net(state)
            mv = max(board.legal_moves, key=lambda m: qvals[0, move_to_idx(m)].item())
            board.push(mv)
            interface.play_sound(mv, board, move_sound, capture_sound, promo_sound)
            turn = chess.WHITE

        interface.draw_board(screen, board, selected, legal_moves)
        pygame.display.flip()
        clock.tick(30)

        if board.is_game_over():
            res = board.result()
            draw_overlay(f"Game Over: {res}", color=(255,0,0))
            pygame.time.wait(2000)
            running = False

def network_play(is_host, arg):
    board = chess.Board()
    recv_queue = queue.Queue()
    stop_event = threading.Event()

    def create_conn():
        while True:
            try:
                if is_host:
                    draw_overlay("Waiting for opponent...")
                    sock = host_game()
                else:
                    draw_overlay("Connecting to host...")
                    sock = join_game(arg)
                sock.settimeout(5.0)
                return sock
            except Exception:
                draw_overlay("Connection failed. Retrying...", color=(255,100,100))
                pygame.time.wait(2000)

    conn = create_conn()

    def listener(sock, q, stop):
        while not stop.is_set():
            try:
                data = sock.recv(BUFFER_SIZE)
                if not data:
                    stop.set(); break
                mv = chess.Move.from_uci(data.decode())
                q.put(mv)
            except socket.timeout:
                continue
            except Exception:
                stop.set(); break

    lt = threading.Thread(target=listener, args=(conn, recv_queue, stop_event), daemon=True)
    lt.start()

    selected = None
    legal_moves = []
    turn = chess.WHITE
    clock = pygame.time.Clock()
    running = True

    while running:
        if stop_event.is_set():
            conn.close()
            conn = create_conn()
            stop_event.clear()
            lt = threading.Thread(target=listener, args=(conn, recv_queue, stop_event), daemon=True)
            lt.start()
            turn = chess.WHITE if is_host else chess.BLACK

        for ev in pygame.event.get():
            if ev.type == pygame.QUIT:
                stop_event.set(); conn.close(); pygame.quit(); sys.exit()
            if ev.type == pygame.MOUSEBUTTONDOWN and turn == (chess.WHITE if is_host else chess.BLACK):
                sq = interface.get_square(ev.pos)
                if selected is not None and sq == selected:
                    selected, legal_moves = None, []
                elif board.piece_at(sq) and board.color_at(sq) == turn:
                    selected = sq
                    legal_moves = [m.to_square for m in board.legal_moves if m.from_square == sq]
                elif selected is not None:
                    move = chess.Move(selected, sq)
                    p = board.piece_at(selected)
                    rank = chess.square_rank(sq)
                    if p and p.piece_type == chess.PAWN and ((turn == chess.WHITE and rank == 7) or (turn == chess.BLACK and rank == 0)):
                        promo = interface.ask_promotion_choice(screen, board)
                        move = chess.Move(selected, sq, promotion=promo)
                    if move in board.legal_moves:
                        board.push(move)
                        try:
                            conn.sendall(move.uci().encode())
                        except Exception:
                            stop_event.set()
                        interface.play_sound(move, board, move_sound, capture_sound, promo_sound)
                        selected, legal_moves = None, []
                        turn = not turn
                    else:
                        selected, legal_moves = None, []

        while not recv_queue.empty():
            mv = recv_queue.get()
            board.push(mv)
            interface.play_sound(mv, board, move_sound, capture_sound, promo_sound)
            turn = chess.WHITE if is_host else chess.BLACK

        interface.draw_board(screen, board, selected, legal_moves)
        pygame.display.flip()
        clock.tick(30)

        if board.is_game_over():
            res = board.result()
            draw_overlay(f"Game Over: {res}", color=(255, 0, 0))
            pygame.time.wait(2000)
            stop_event.set(); conn.close(); running = False

def ask_host_join():
    font = pygame.font.Font(None, 40)
    clock = pygame.time.Clock()
    hb = pygame.Rect(WIDTH//2 - 150, HEIGHT//2 - 25, 120, 50)
    jb = pygame.Rect(WIDTH//2 + 30, HEIGHT//2 - 25, 120, 50)
    ib = pygame.Rect(WIDTH//2 - 100, HEIGHT//2 + 50, 200, 40)
    active = False
    text = ''
    while True:
        screen.fill((0, 0, 0))
        for ev in pygame.event.get():
            if ev.type == pygame.QUIT:
                pygame.quit(); sys.exit()
            if ev.type == pygame.MOUSEBUTTONDOWN:
                if hb.collidepoint(ev.pos):
                    network_play(True, None)
                    return
                if jb.collidepoint(ev.pos):
                    active = True
            if ev.type == pygame.KEYDOWN and active:
                if ev.key == pygame.K_RETURN:
                    network_play(False, text)
                    return
                if ev.key == pygame.K_BACKSPACE:
                    text = text[:-1]
                else:
                    text += ev.unicode

        pygame.draw.rect(screen, (0, 255, 0), hb)
        pygame.draw.rect(screen, (255, 0, 255), jb)
        screen.blit(font.render("Host", True, (0, 0, 0)), (hb.x + 20, hb.y + 10))
        screen.blit(font.render("Join", True, (0, 0, 0)), (jb.x + 20, jb.y + 10))
        if active:
            surf = font.render(text, True, (255, 255, 255))
            ib.w = max(200, surf.get_width() + 10)
            screen.blit(surf, (ib.x + 5, ib.y + 5))
            pygame.draw.rect(screen, (30, 144, 255), ib, 2)

        pygame.display.flip(); clock.tick(30)

def mode_menu():
    font = pygame.font.Font(None, 50)
    clock = pygame.time.Clock()
    lb = pygame.Rect(150, HEIGHT//2 - 25, 140, 50)
    ob = pygame.Rect(310, HEIGHT//2 - 25, 140, 50)
    while True:
        screen.fill((30, 30, 30))
        for ev in pygame.event.get():
            if ev.type == pygame.QUIT:
                pygame.quit(); sys.exit()
            if ev.type == pygame.MOUSEBUTTONDOWN:
                if lb.collidepoint(ev.pos): local_play(); return
                if ob.collidepoint(ev.pos): ask_host_join(); return
        pygame.draw.rect(screen, (0, 200, 0), lb)
        screen.blit(font.render("Local", True, (255, 255, 255)), (lb.x + 40, lb.y + 10))
        pygame.display.flip(); clock.tick(30)

def main_menu():
    font = pygame.font.Font(None, 50)
    clock = pygame.time.Clock()
    pb = pygame.Rect((WIDTH - 200)//2, (HEIGHT - 50)//2, 200, 50)
    while True:
        screen.fill((0, 0, 0))
        for ev in pygame.event.get():
            if ev.type == pygame.QUIT:
                pygame.quit(); sys.exit()
            if ev.type == pygame.MOUSEBUTTONDOWN:
                if pb.collidepoint(ev.pos): mode_menu()
        pygame.draw.rect(screen, (0, 128, 255), pb)
        screen.blit(font.render("Play", True, (255, 255, 255)), (pb.x + 60, pb.y + 10))
        pygame.display.flip(); clock.tick(30)