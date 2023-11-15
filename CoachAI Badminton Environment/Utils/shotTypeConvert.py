"""
Convert shot type name from English to Traditional Chinese and back forth
"""

shot_type_zh2en_mapping = {'發短球': 'short service', '長球': 'clear', 
                            '推撲球': 'push/rush', '殺球':'smash', '接殺防守': 'defensive shot',
                            '平球': 'drive', '網前球': 'net shot', '挑球': 'lob', 
                            '切球': 'drop', '發長球': 'long service','推球':'push','擋小球': 'defensive shot','放小球': 'net shot'}

shot_type_en2zh_mapping = {'short service':'發短球', 'clear':'長球', 
                            'push/rush':'推撲球','smash': '殺球', 'defensive shot':'接殺防守',
                            'drive':'平球', 'net shot':'網前球', 'lob':'挑球', 
                            'drop':'切球', 'long service':'發長球','push':'推球'}
# ['發短球', '長球', '推撲球', '殺球', '接殺防守', '平球', '網前球', '挑球', '切球', '發長球']
def shotTypeZh2En(zh: str):
    if zh in shot_type_zh2en_mapping:
        return shot_type_zh2en_mapping[zh]
    else:
        return "unknown"

def shotTypeEn2Zh(en: str):
    if en in shot_type_en2zh_mapping:
        return shot_type_en2zh_mapping[en]
    else:
        return "未知球種"