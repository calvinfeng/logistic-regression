import json

def get_matches():
    json_data = open('matches.json')
    matches = json.load(json_data)['matches']
    return matches # which is an array of matches

def get_team_gold_rate(team, match_duration):
    total_gold = 0
    for teammate in team:
        total_gold += teammate['stats']['goldEarned']
    return float(total_gold)/match_duration

def get_team_kda(team):
    kills, deaths, assists = 0, 0, 0
    for teammate in team:
        kills += teammate['stats']['kills']
        deaths += teammate['stats']['deaths']
        assists += teammate['stats']['assists']
    return float(kills + assists)/deaths

def get_team_cs_rate(team):
    cs_per_min = 0
    for teammate in team:
        cs_deltas = teammate['timeline']['creepsPerMinDeltas']
        cs_per_min += float(cs_deltas['zeroToTen'] + cs_deltas['tenToTwenty'])/2
    return cs_per_min

def get_parsed_data():
    parsed_data = []
    matches = get_matches()
    for match in matches:
        team_1_data, team_2_data = {}, {}
        team_1_data['x'], team_2_data['x'] = [], []

        team_1 = match['participants'][0:5]
        team_1_data['x'].append(1)
        team_1_data['x'].append(get_team_gold_rate(team_1, match['matchDuration']))
        team_1_data['x'].append(get_team_kda(team_1))
        team_1_data['x'].append(get_team_cs_rate(team_1))

        team_2 = match['participants'][5:10]
        team_2_data['x'].append(1)
        team_2_data['x'].append(get_team_gold_rate(team_2, match['matchDuration']))
        team_2_data['x'].append(get_team_kda(team_2))
        team_2_data['x'].append(get_team_cs_rate(team_2))

        if match['teams'][0]['winner']:
            team_1_data['y'] = 1
            team_2_data['y'] = 0
        else:
            team_1_data['y'] = 0
            team_2_data['y'] = 1
        parsed_data.append(team_1_data)
        parsed_data.append(team_2_data)
    return parsed_data
