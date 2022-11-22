import numpy as np
import pandas as pd
import math

def velocity(time_current,time_next,hit_x,hit_y,landing_x,landing_y):
    
    if type(time_current) == float:
        return('0')

    hhmm1 = time_current.split(':',2)
    if len(hhmm1) == 2:
        ms1 = int(hhmm1[0])*60*1000+float(hhmm1[1])*1000
    else:
        ms1 = int(hhmm1[0])*60*60*1000+int(hhmm1[1])*60*1000+float(hhmm1[2])*1000
    
    hhmm2 = time_next.split(':',2)
    if len(hhmm2) == 2:
        ms2 = int(hhmm2[0])*60*1000+float(hhmm2[1])*1000
    else:
        ms2 = int(hhmm2[0])*60*60*1000+int(hhmm2[1])*60*1000+float(hhmm2[2])*1000
    
    ms = abs(ms2-ms1)

    x_sqaure = pow(landing_x - hit_x,2)
    y_sqaure = pow(landing_y - hit_y,2)
    distance = pow(y_sqaure+x_sqaure,1/2)


    velocity = distance/70/1000/(ms/1000/60/60) # km/h

    return round(velocity,3)

def direction(diagonal_angle,hit_x,hit_y,hit_area,landing_x,landing_y,landing_area):

    if type(hit_area) == float:
        return('')
    if (hit_area[0] == 'C' or hit_area[0] == 'E') and \
    (landing_area[0] == 'C' or landing_area[0] == 'E'):
        return 2
    if (landing_area[0] == 'B' or landing_area[0] == 'D') and \
    (hit_area[0] == 'B' or hit_area[0] == 'D'):
        return 2
    if landing_y > hit_y:
        compare_y = 0
    else:
        compare_y = 935
    
    x1 = landing_x - hit_x
    x2 = 0
    y1 = landing_y - hit_y
    y2 = compare_y - hit_y

    dot = x1*x2+y1*y2
    distance1 = pow(x1*x1+y1*y1,1/2)
    distance2 = pow(x2*x2+y2*y2,1/2)

    cos_angle = abs(dot/(distance1*distance2))
    
    if cos_angle < math.cos(diagonal_angle/360*2*math.pi):
        return 2
    else:
        return 1


def ball_type_convertion(ball_type):
    if ball_type == '切球' or ball_type == '過渡切球':
        return 'cut'
    elif ball_type == '平球' or ball_type == '小平球' or ball_type == '後場抽平球':
        return 'drive'
    elif ball_type == '挑球' or ball_type == '防守回挑':
        return 'lob'
    elif ball_type == '長球' or ball_type == '發長球':
        return 'long'
    elif ball_type == '發小球' or ball_type == '放小球' or ball_type == '擋小球':
        return 'netplay'
    elif ball_type == '撲球':
        return 'rush'
    elif ball_type == '殺球':
        return 'smash'
    else:
        return 'error'


def hit_convertion_9(hit):
    if hit[0] == 'A':
        return '2',hit[1]
    elif hit[0] == 'B':
        return '3',hit[1]
    elif hit[0] == 'C':
        return '1',hit[1]
    elif hit[0] == 'D':
        return '4',hit[1]
    elif hit[0] == 'E':
        return '0',hit[1]
    else:
        return 'X'

def landing_convertion_9(landing):
    if landing[0] == 'A':
        return '2',landing[1]
    elif landing[0] == 'B':
        return '1',landing[1]
    elif landing[0] == 'C':
        return '3',landing[1]
    elif landing[0] == 'D':
        return '0',landing[1]
    elif landing[0] == 'E':
        return '4',landing[1]
    else:
        return 'X'
    
def hit_convertion(hit):
    if hit[0] == 'A':
        return '2',hit[1]
    elif hit[0] == 'B':
        return '3',hit[1]
    elif hit[0] == 'C':
        return '1',hit[1]
    elif hit[0] == 'D':
        return '4',hit[1]
    elif hit[0] == 'E':
        return '0',hit[1]
    elif hit[0] == 'F':
        return '5',hit[1]
    else:
        return 'X'

def landing_convertion(landing):
    if landing[0] == 'A':
        return '3',landing[1]
    elif landing[0] == 'B':
        return '2',landing[1]
    elif landing[0] == 'C':
        return '4',landing[1]
    elif landing[0] == 'D':
        return '1',landing[1]
    elif landing[0] == 'E':
        return '5',landing[1]
    elif landing[0] == 'F':
        return '0',landing[1]
    else:
        return 'X'

def map_reason(reason):
    if reason == 0:
        return '出界'
    elif reason == 1:
        return '落地'
    elif reason == 2:
        return '未回擊成功'
    else:
        return ''
def revese_map_reason(reason_name):
    if reason_name == '出界':
        return 0
    elif reason_name == '落地':
        return 1
    elif reason_name == '未回擊成功':
        return 2
    else:
        return ''

def another_player(player):
    if player == 'A':
        return 'B'
    elif player == 'B':
        return 'A'
    else:
        return ''

def who_first_blood(reason_name, winner):
    reason = revese_map_reason(reason_name)
    if reason == 0 or reason == 2:
        return another_player(winner)
    elif reason == 1:
        return winner