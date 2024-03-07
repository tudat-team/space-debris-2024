# implementation according to https://www.youtube.com/watch?v=EM1s8jHa0L8&list=PLBLV84VG7Md8SgHlXuQXPMJLDvCaWVVQv

import pygame
import random
import math 
from engine import Creature

pygame.init()

# global variables
CELL_NUMBER = 8
SQ_SIZE = 50
FOV_HEIGHT = 3
FOV_WIDTH = 3
MAX_NUM_ROUNDS = 15
PENELTY_MISSING = 10
PENELTY_TOO_MANY = 5
PIXEL_OFFSET = 20

WIDTH, HEIGHT = SQ_SIZE * CELL_NUMBER, SQ_SIZE * CELL_NUMBER
WIN = pygame.display.set_mode((WIDTH + 500, HEIGHT+200))
pygame.display.set_caption("Zombieland")

# buttons
font = pygame.font.SysFont('agencyfb', 25, bold=True)
font2 = pygame.font.SysFont('agencyfb', 40, bold=True)
button_light_on_off = pygame.Rect(575, 400, 140, 60)
flashlight = pygame.image.load('flashlight.png')
flashlight_small = pygame.transform.scale(flashlight, (100, 100))
flashlight = pygame.transform.scale(flashlight, (150, 150))
button_flashlight_small = flashlight_small.get_rect()
button_flashlight = flashlight.get_rect()
replay = pygame.image.load('play_again.png')
replay = pygame.transform.scale(replay, (150, 150))
button_replay = replay.get_rect()

# cursor
targeting_cursor = pygame.cursors.Cursor(pygame.cursors.broken_x)
default_cursor = pygame.cursors.Cursor(pygame.cursors.arrow)

# colours
GREY = (40, 50, 60)
BLACK = (0, 0, 0)
GREEN = (50, 200, 150)
YELLOW = (250, 250, 0)
WHITE = (255, 255, 255)
MONITOR_GREEN = (175, 225, 175)


def draw_evaluation_window():
    # draw background
    WIN.fill(GREY)   

    # draw grid
    draw_grid()

    # print information
    text = font.render('Click on <Register Creature> ', True, (255,255,255))
    text1 = font.render('and mark the position of ', True, (255,255,255))
    text2 = font.render('all creatures in the grid.', True, (255,255,255))
    text3 = font.render('Then click on <Submit>.', True, (255,255,255))

    WIN.blit(text, (530, 50))
    WIN.blit(text1, (530, 75))
    WIN.blit(text2, (530, 100))
    WIN.blit(text3, (530, 125))

    # draw registration button zombies
    button_register_zombie = pygame.Rect(510, 230, 160, 80)
    text_register_zombie = font.render('Register Zombies', True, WHITE)
    pygame.draw.rect(WIN, GREEN, button_register_zombie)
    WIN.blit(text_register_zombie, (515, 250))

    # draw submit button zombies
    button_submit_zombies = pygame.Rect(695, 230, 70, 80)
    text_submit = font.render('Submit', True, WHITE)
    pygame.draw.rect(WIN, GREEN, button_submit_zombies)
    WIN.blit(text_submit, (700, 250))

    # draw registration button humans
    button_register_human = pygame.Rect(510, 330, 160, 80)
    text_register_human = font.render('Register Humans', True, (0,0,0))
    pygame.draw.rect(WIN, YELLOW, button_register_human)
    WIN.blit(text_register_human, (515, 350))

    # draw submit button humans
    button_submit_humans = pygame.Rect(695, 330, 70, 80)
    text_submit = font.render('Submit', True, BLACK)
    pygame.draw.rect(WIN, YELLOW, button_submit_humans)
    WIN.blit(text_submit, (700, 350))

    # draw clear button
    button_clear = pygame.Rect(610, 430, 100, 40)
    text_clear = font.render('Clear', True, BLACK)
    pygame.draw.rect(WIN, WHITE, button_clear)
    WIN.blit(text_clear, (630, 435))
    return button_register_zombie, button_register_human, button_clear, button_submit_zombies, button_submit_humans

def display_result(score_total, registered_zombies, registered_humans):

    # draw background
    WIN.fill(GREY)   

    # draw grid
    draw_grid()

    # draw true position of zombies and humans
    for zombie in allZombies:
        draw_zombies(zombie, True)
    for human in allHumans:
        draw_humans(human, True)

    # draw registered position of zombies and humans
    for zombie in registered_zombies:
        adjust_object_on_grid(zombie.col * SQ_SIZE, zombie.row * SQ_SIZE, False, GREEN)
    for human in registered_humans:
        adjust_object_on_grid(human.col * SQ_SIZE, human.row * SQ_SIZE, False, YELLOW)

    # display result
    text_result = font2.render('You scored ', True, WHITE)
    text_score = font2.render(str(score_total), True, WHITE)
    WIN.blit(text_result, (550, 170))
    WIN.blit(text_score, (700, 175))

    # draw play again button
    button_play_again = pygame.Rect(550, 280, 170, 170)
    text__play_again = font.render('Play again', True, BLACK)
    pygame.draw.rect(WIN, GREY, button_play_again)
    WIN.blit(text__play_again, (610, 425))

    button_replay.x = 570
    button_replay.y = 270
    WIN.blit(replay, button_replay)

    return button_replay, button_play_again

    # track user interaction
    # while True:
    #     for event in pygame.event.get():
    #         # user closes the pygame window
    #         if event.type == pygame.QUIT:
    #             return False
    #         if event.type == pygame.MOUSEBUTTONDOWN:
    #             if event.button == 1:   # left mouse click
    #                 if button_play_again.collidepoint(event.pos):
    #                     print('Play again')
    #                     return True
    

# Initialise endgame
def evaluate():
        # print position of all true zombies
        print('Index true zombies: ')
        for zombie in allZombies:
            print(zombie.index)

        registered_zombies = set()
        registered_humans = set()
        registering_zombies = False
        registering_humans = False
        first_creature = False
        submit_zombies = False
        submit_humans = False
        run_evaluate = True
        make_creature_visible = False
        place_creature = False
        loc_creature = []
        clear_disabled = False
        play_again_disabled = True
        button_replay = pygame.Rect(0, 0, 0, 0)
        button_play_again = pygame.Rect(615, 430, 75, 40)

        button_register_zombie, button_register_human, button_clear, button_submit_zombies, button_submit_humans = draw_evaluation_window()
        
        while run_evaluate:
            caption = pygame.display.get_caption()
            if caption[0] == "Zombieland - Result":
                play_again_disabled = False
            else:
                play_again_disabled = True

            if caption[0] == "Zombieland - Endgame":
                clear_disabled = False
            else:
                clear_disabled = True

            # track user interaction
            for event in pygame.event.get():
                # user closes the pygame window
                if event.type == pygame.QUIT:
                    run_evaluate = False
                    return False
                if event.type == pygame.MOUSEBUTTONDOWN:
                    if event.button == 1:   # left mouse click
                        # print('clear disabled', clear_disabled)
                        # print('play again disabled', play_again_disabled)
                        if button_replay.collidepoint(event.pos) or button_play_again.collidepoint(event.pos):
                            if play_again_disabled:
                                break
                            
                            # # clear_disabled = False
                            # play_again_disabled = True
                            return True
                            
                        if place_creature:
                            make_creature_visible = True
                            
                        if button_register_zombie.collidepoint(event.pos):
                            # register all positions of zombies
                            if registering_humans or submit_zombies:
                                break
                            pygame.mouse.set_cursor(targeting_cursor)
                            registering_zombies = True
                            registering_humans = False
                            place_creature = True
                            first_creature = True

                        elif button_register_human.collidepoint(event.pos):
                            # register all positions of humans
                            if registering_zombies or submit_humans:
                                break
                            pygame.mouse.set_cursor(targeting_cursor)
                            print('num zombies:', len(registered_zombies))
                            print('num humans:', len(registered_humans))
                            registering_zombies = False
                            registering_humans = True
                            place_creature = True
                            first_creature = True

                        if button_clear.collidepoint(event.pos):
                            if clear_disabled:
                                break
                            pygame.mouse.set_cursor(default_cursor)
                            registered_zombies.clear()
                            registered_humans.clear()
                            registering_zombies = False
                            registering_humans = False
                            first_creature = False
                            submit_zombies = False
                            submit_humans = False
                            run_evaluate = True
                            make_creature_visible = False
                            place_creature = False
                            loc_creature = []
                            draw_evaluation_window()
                            clear_disabled = False
                            pygame.display.flip()
                            pygame.display.update()
                            print('num zombies: ', len(registered_zombies))
                            print('num humans: ', len(registered_humans))

                        if button_submit_zombies.collidepoint(event.pos):
                            if registering_humans:
                                break
                            pygame.mouse.set_cursor(default_cursor)
                            submit_zombies = True
                            place_creature = False
                            registering_zombies = False
                        if button_submit_humans.collidepoint(event.pos):
                            if registering_zombies:
                                break
                            pygame.mouse.set_cursor(default_cursor)
                            submit_humans = True
                            place_creature = False
                            registering_humans = False
                            print('num zombies: ', registered_zombies)
                            print('num humans: ', registered_humans)

                        if first_creature:
                            first_creature = False
                        else:    
                            if registering_zombies:
                                color_frame_creature = GREEN
                            elif registering_humans:
                                color_frame_creature = YELLOW
                            to_draw = draw_object_on_grid(make_creature_visible)
                            if len(to_draw) == 2:
                                loc_creature = (to_draw[0], to_draw[1])
                            if len(loc_creature) != 0:
                                row, col, size = adjust_object_on_grid(loc_creature[0], loc_creature[1], False, color_frame_creature)
                                if registering_zombies:
                                    registered_zombies.add(Creature(CELL_NUMBER, row, col, 'Zombie'))
                                elif registering_humans:
                                    registered_humans.add(Creature(CELL_NUMBER, row, col, 'Human'))
                                make_creature_visible = False
                        print('num zombies: ', len(registered_zombies))
                        print('num humans: ', len(registered_humans))

                        # Compute score
                        if submit_zombies and submit_humans:
                            clear_disabled = True
                            # check zombies first
                            score_zombie = computeScore(True, registered_zombies)
                            # check humans
                            score_humans = computeScore(False, registered_humans)

                            score_total = round(score_zombie + score_humans)

                            # Display score
                            pygame.display.set_caption("Zombieland - Result")
                            button_replay, button_play_again = display_result(score_total, registered_zombies, registered_humans)

            pygame.display.update()                
                            
                
# draw grid
def draw_grid(left = 0, top = 0):
    for i in range(CELL_NUMBER*CELL_NUMBER):
        x = i % CELL_NUMBER * SQ_SIZE
        y = i // CELL_NUMBER * SQ_SIZE
        square = pygame.Rect(x, y, SQ_SIZE, SQ_SIZE)
        pygame.draw.rect(WIN, BLACK, square, width=3)
    
    # print labels
        alpha = [chr(i) for i in range(ord('a'), ord('z')+1)]
    row_letters = CELL_NUMBER * SQ_SIZE + 10
    col_numbers = CELL_NUMBER * SQ_SIZE + 10
    for i in range(CELL_NUMBER):
        text = font.render(str(i), True, (255,255,255))
        WIN.blit(text, (i*SQ_SIZE + (SQ_SIZE/2), row_letters))

        text2 = font.render(alpha[i], True, (255,255,255))
        WIN.blit(text2, (col_numbers, i*SQ_SIZE + 10))
    return

# draw zombies onto position grids
def draw_zombies(zombie, display_result):
    x = zombie.col * SQ_SIZE + (PIXEL_OFFSET/2)
    y = zombie.row * SQ_SIZE + (PIXEL_OFFSET/2)
    rectangle = pygame.Rect(x, y, SQ_SIZE-PIXEL_OFFSET, SQ_SIZE-PIXEL_OFFSET)
    if display_result:
        pygame.draw.rect(WIN, GREEN, rectangle)
    else:
        pygame.draw.rect(WIN, GREY, rectangle)

def draw_humans(human, display_result):
    x = human.col * SQ_SIZE + (PIXEL_OFFSET/2)
    y = human.row * SQ_SIZE + (PIXEL_OFFSET/2)
    rectangle = pygame.Rect(x, y, SQ_SIZE-PIXEL_OFFSET, SQ_SIZE-PIXEL_OFFSET)
    if display_result:
        pygame.draw.rect(WIN, YELLOW, rectangle)
    else:
        pygame.draw.rect(WIN, GREY, rectangle)

def draw_light_button(alreadyPressed):
    if alreadyPressed:
        text = font.render('Light on!', True, (255,255,255))
        pygame.draw.rect(WIN, (255, 168, 54), button_light_on_off)
        WIN.blit(text, (590, 415))
    else:
        text = font.render('Light off!', True, (255,255,255))
        pygame.draw.rect(WIN, (136, 128, 123), button_light_on_off)
        WIN.blit(text, (590, 415))

def draw_flashlight(useFlashlight):
    if not(useFlashlight):
        button_flashlight.x = 500
        button_flashlight.y = 200
        WIN.blit(flashlight, button_flashlight)

def draw_flashlight_small(useFlashlight):
    if not(useFlashlight):
        button_flashlight_small.x = 660
        button_flashlight_small.y = 230
        WIN.blit(flashlight_small, button_flashlight_small)

def draw_monitor(numCreatures, numZombies, numHumans, light_off):
    text = font.render('Number of Detections', True, (255,255,255))
    text_descrip_zombies = font.render('Zombies: ', True, (255,255,255))
    text_descrip_humans = font.render('Humans: ', True, (255,255,255))

    text_numDetections = font2.render(str(numCreatures), True, (255,255,255))
    text_numZombies = font.render(str(numZombies), True, (255,255,255))
    text_numHumans = font.render(str(numHumans), True, (255,255,255))

    monitor_frame = pygame.Rect(540, 40, 220, 170)
    monitor = pygame.Rect(550, 50, 200, 150)
    pygame.draw.rect(WIN, BLACK, monitor_frame)
    pygame.draw.rect(WIN, MONITOR_GREEN, monitor)
    WIN.blit(text, (560, 50))
    WIN.blit(text_descrip_zombies, (560, 130))
    WIN.blit(text_descrip_humans, (660, 130))


    if not(light_off):
        WIN.blit(text_numDetections, (640, 80))
        WIN.blit(text_numZombies, (585, 160))
        WIN.blit(text_numHumans, (685, 160))    

def draw_object_on_grid(placeFov):
    if placeFov:
        x, y = pygame.mouse.get_pos()
        if x<SQ_SIZE*CELL_NUMBER:
            return (x, y)
        else:
            return tuple()
    else:
        return tuple()
    
def adjust_object_on_grid(x_to_Adjust, y_to_Adjust, isLargeFov, colour):
    col = x_to_Adjust // SQ_SIZE
    row = y_to_Adjust // SQ_SIZE
    x = col  * SQ_SIZE
    y = row  * SQ_SIZE

    # check if FOV is at right boundary of the grid
    if not(isLargeFov):
        fov_size = 1
        rectangle = pygame.Rect(x, y, fov_size*SQ_SIZE, fov_size*SQ_SIZE)
        pygame.draw.rect(WIN, colour, rectangle, width=2)
        return (row, col, fov_size) # for small FOV use size of one
    
    adjusted_width_fov = CELL_NUMBER-col
    adjusted_height_fov = CELL_NUMBER-row
    if adjusted_width_fov<FOV_WIDTH or adjusted_height_fov<FOV_WIDTH:
        rectangle = pygame.Rect(x, y, adjusted_width_fov*SQ_SIZE, adjusted_height_fov*SQ_SIZE)
        pygame.draw.rect(WIN, colour, rectangle, width=2)
    else:
        adjusted_width_fov = FOV_WIDTH
        rectangle = pygame.Rect(x, y, FOV_WIDTH*SQ_SIZE, FOV_HEIGHT*SQ_SIZE)
        pygame.draw.rect(WIN, colour, rectangle, width=2)
    return (row, col, adjusted_width_fov)

def computeScore(checkZombies, estimated_creatures):
    cost = 0
    if checkZombies:
        truth = allZombies.copy()
    else:
        truth = allHumans.copy()
    estimated = estimated_creatures.copy()

    if len(estimated_creatures) < len(truth):
        # registered too less creatures
        cost = cost - PENELTY_MISSING * abs(len(truth) - len(estimated_creatures))

        for estimated_creature in estimated:
            min_dist = float('inf')
            closest_creature = -1
            
            # retrieve position of registered creature
            estimated_row = estimated_creature.row
            estimated_col = estimated_creature.col
            
            for creature in truth:
                # Compute distance between true zombie with registered one
                dist = math.sqrt(math.pow(estimated_row - creature.row, 2) + math.pow(estimated_col - creature.col, 2))
                if dist<min_dist:
                    min_dist = dist
                    closest_creature = creature
            
            # re-compute cost
            cost = cost - min_dist

            # remove registered creature 
            truth.remove(closest_creature)

        return cost
        
    elif len(truth) < len(estimated_creatures):
        # registered too many creatures
        cost = cost - PENELTY_TOO_MANY * abs(len(truth) - len(estimated_creatures))
    
    # registered correct number 
    for creature in truth:
        min_dist = float('inf')
        closest_creature = -1
        for estimated_creature in estimated:

            # retrieve position of registered creature
            estimated_row = estimated_creature.row
            estimated_col = estimated_creature.col

            # Compute distance between true zombie with registered one
            dist = math.sqrt(math.pow(estimated_row - creature.row, 2) + math.pow(estimated_col - creature.col, 2))
            if dist<min_dist:
                min_dist = dist
                closest_creature = estimated_creature
        
        # re-compute cost
        cost = cost - min_dist

        # remove registered creature 
        estimated.remove(closest_creature)
    
    return cost
    

# genrate set of zombies
allZombies = set()
maxNumZombies = 5
numZombies = random.randrange(3, maxNumZombies)
for i in range(numZombies):
    row = random.randrange(0, CELL_NUMBER)
    col = random.randrange(0, CELL_NUMBER)
    zombie = Creature(CELL_NUMBER, row, col, 'Zombie')
    allZombies.add(zombie)

# generate set of humans
allHumans = set()
maxNumHumans = 5
numHumans = random.randrange(3, maxNumHumans)
for i in range(numHumans):
    row = random.randrange(0, CELL_NUMBER)
    col = random.randrange(0, CELL_NUMBER)
    human = Creature(CELL_NUMBER, row, col, 'Human')
    allHumans.add(human)

# execute game
def main():
    counter_rounds = 1
    run = True
    pausing = False
    light_off = True
    use_flashlight = False
    use_flashlight_small = False
    place_fov = False
    make_fov_visible = False
    place_fov_small = False
    make_fov_visible_small = False
    loc_fov = []
    loc_fov_small = []
    num = 0
    registered_zombies = 0
    registered_humans = 0

    while run:
   
        # track user interaction
        for event in pygame.event.get():

            # user closes the pygame window
            if event.type == pygame.QUIT:
                run = False
                break
        
            if event.type == pygame.MOUSEBUTTONDOWN:
                pygame.mouse.set_cursor(default_cursor)
                if event.button == 1:   # left mouse click
                    if place_fov:
                        make_fov_visible = True
                        place_fov = False
                        break
                    if place_fov_small:
                        make_fov_visible_small = True
                        place_fov_small = False
                        break
                    if light_off:
                        if button_light_on_off.collidepoint(event.pos):
                            light_off = False
                            place_fov = False
                            place_fov_small = False
                            
                        # player is supposed to place the FOV
                        if button_flashlight.collidepoint(event.pos):
                            use_flashlight = True   # flashlight currently in use
                            pygame.mouse.set_cursor(targeting_cursor)
                            place_fov = True

                        if button_flashlight_small.collidepoint(event.pos):
                            use_flashlight_small = True   # flashlight currently in use
                            pygame.mouse.set_cursor(targeting_cursor)
                            place_fov_small = True
                    else:
                        place_fov = False
                        place_fov_small = False
                        if button_light_on_off.collidepoint(event.pos):
                            light_off = True
                            use_flashlight = False  # flashlight can only be used when light is off
                            use_flashlight_small = False
                            loc_fov = []
                            loc_fov_small = []
                            counter_rounds = counter_rounds + 1
                            if counter_rounds < MAX_NUM_ROUNDS:
                                # Creatures are moving
                                # store positional indicies of zombies in a set
                                zombies_indicies = set([])

                                # move creatures
                                for zombie in allZombies:
                                    zombie.move(CELL_NUMBER, 'Zombie')
                                    zombies_indicies.add(zombie.index)

                                humans_2_zombies = set()
                                for human in allHumans:
                                    human.move(CELL_NUMBER, 'Human', zombies_indicies)
                                    if human.type == 'Zombie':
                                        allZombies.add(human)
                                        humans_2_zombies.add(human)

                                allHumans.difference_update(humans_2_zombies)

                elif event.button == 3 and light_off == True: # right mouse click
                    use_flashlight = False
                    use_flashlight_small = False
                    #place_fov = False


            # user presses key on keyboard
            if event.type == pygame.KEYDOWN:

                # escape key to close the window
                if event.key == pygame.K_ESCAPE:
                    run = False


        if not pausing:

            # draw background
            WIN.fill(GREY)   
        
            # draw grid
            draw_grid()

            draw_light_button(light_off)

            draw_flashlight(use_flashlight)

            draw_flashlight_small(use_flashlight_small)

            # place Fov of flashlight
            to_draw = draw_object_on_grid(make_fov_visible)
            to_draw_small = draw_object_on_grid(make_fov_visible_small)

            if len(to_draw) == 2:
                loc_fov = [to_draw[0], to_draw[1]]
            
            if len(to_draw_small) == 2:
                loc_fov_small = [to_draw_small[0], to_draw_small[1]]

            if len(loc_fov) != 0:
                row, col, adjusted_width_fov = adjust_object_on_grid(loc_fov[0], loc_fov[1], True, WHITE) # for large flashlight

                # count how many creatures in large FOV
                num_creatures_in_fov = 0
                for r in range(FOV_HEIGHT):
                    for c in range(adjusted_width_fov):
                        fov_index = (row+r) * CELL_NUMBER + (col+c)
                        for zombie in allZombies:
                            if zombie.get_index(CELL_NUMBER) == fov_index:
                                num_creatures_in_fov = num_creatures_in_fov + 1
                        for human in allHumans:
                            if human.get_index(CELL_NUMBER) == fov_index:
                                num_creatures_in_fov = num_creatures_in_fov + 1
                num = num_creatures_in_fov
            
            if len(loc_fov_small) != 0:
                row, col, adjusted_width_fov = adjust_object_on_grid(loc_fov_small[0], loc_fov_small[1], False, WHITE) # for small flashlight
                fov_index = row * CELL_NUMBER + col
                registered_zombies = 0
                for zombie in allZombies:
                    if zombie.get_index(CELL_NUMBER) == fov_index:
                        registered_zombies = registered_zombies + 1
                registered_humans = 0
                for human in allHumans:
                    if human.get_index(CELL_NUMBER) == fov_index:
                        registered_humans = registered_humans + 1
            make_fov_visible = False 
            make_fov_visible_small = False

            # draw zombies
            for zombie in allZombies:
                draw_zombies(zombie, False)

            for human in allHumans:
                draw_humans(human, False)
        
            draw_monitor(num, registered_zombies, registered_humans, light_off)
            pygame.display.flip()
            pygame.display.update()

            if counter_rounds > MAX_NUM_ROUNDS:
                pygame.display.set_caption("Zombieland - Endgame")
                continue_running = evaluate()
                if not(continue_running):
                    run = False
                    break
                else:

                    # reset and play new round
                    pygame.display.set_caption("Zombieland")
                    counter_rounds = 1
                    run = True
                    pausing = False
                    light_off = True
                    use_flashlight = False
                    use_flashlight_small = False
                    place_fov = False
                    make_fov_visible = False
                    place_fov_small = False
                    make_fov_visible_small = False
                    loc_fov = []
                    loc_fov_small = []
                    num = 0
                    registered_zombies = 0
                    registered_humans = 0

                    # genrate new set of zombies
                    allZombies.clear()
                    maxNumZombies = 5
                    numZombies = random.randrange(1, maxNumZombies)
                    for i in range(numZombies):
                        row = random.randrange(0, CELL_NUMBER)
                        col = random.randrange(0, CELL_NUMBER)
                        zombie = Creature(CELL_NUMBER, row, col, 'Zombie')
                        allZombies.add(zombie)

                    # generate new set of humans
                    allHumans.clear()
                    maxNumHumans = 5
                    numHumans = random.randrange(1, maxNumHumans)
                    for i in range(numHumans):
                        row = random.randrange(0, CELL_NUMBER)
                        col = random.randrange(0, CELL_NUMBER)
                        human = Creature(CELL_NUMBER, row, col, 'Human')
                        allHumans.add(human)

    pygame.quit()

if __name__ == "__main__":
    main()