import pygame
pygame.init()
screen = pygame.display.set_mode((600, 600))
clock = pygame.time.Clock()

counter, text = 0, '0'.center(25)
text1 = 'Key Legends:'
text2 = 'Enter = Reset'
text3 = 'Space = Start/Resume/Pause'
text4 = 'Esc = Stop/Exit'

pygame.time.set_timer(pygame.USEREVENT, 1000)
font = pygame.font.SysFont('Consolas', 40)
font2 = pygame.font.SysFont('Consolas', 18)

Count_ongoing = False

while True:
    for event in pygame.event.get():
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_SPACE:
                if Count_ongoing == False:
                    Count_ongoing = True
                elif Count_ongoing == True:
                    Count_ongoing = False
            if event.key == pygame.K_RETURN:
                counter = 0
                text = str(counter).center(25)
            if event.key == pygame.K_ESCAPE:
                break

        if event.type == pygame.USEREVENT and Count_ongoing == True: 
            counter += 1
            text = str(counter).center(25)
        if event.type == pygame.QUIT: break
    else:
        screen.fill((135, 206, 235))
        screen.blit(font.render(text, True, (0, 0, 0)), (32, 48))
        screen.blit(font2.render(text1, True, (0, 0, 0)),(5,500))
        screen.blit(font2.render(text2, True, (0, 0, 0)),(5,515))
        screen.blit(font2.render(text3, True, (0, 0, 0)),(5,530))
        screen.blit(font2.render(text4, True, (0, 0, 0)),(5,545))
        pygame.display.flip()
        clock.tick(60)
        continue
    break