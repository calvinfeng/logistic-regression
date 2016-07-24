require 'json'
require 'byebug'
# Return an array of matches
class MatchLoader

  def self.get_matches
    file = File.read('matches.json')
    JSON.parse(file)['matches']
  end

  def self.get_team_gold_earned(participants)
    total_gold = 0
    participants.each do |participant|
      total_gold += participant['stats']['goldEarned']
    end
    total_gold
  end

  def self.get_team_kda(participants)
    kills, deaths, assists = 0, 0, 0
    participants.each do |participant|
      kills += participant['stats']['kills']
      deaths += participant['stats']['deaths']
      assists += participant['stats']['assists']
    end
    (kills + assists).to_f/deaths
  end

  def self.get_team_cs_rate(participants)
    cs_per_min = 0
    participants.each do |participant|
      cs_deltas = participant['timeline']['creepsPerMinDeltas']
      cs_per_min += (cs_deltas['zeroToTen'] + cs_deltas['tenToTwenty'])/2
    end
    cs_per_min
  end

  # Data format
  # Each data point is a hash as {x => [x1, x2, x3], y => 0 or 1}
  # Store all data points in an array
  def self.parse_data
    data = []
    matches = get_matches
    matches.each do |match|
      t1_dp, t2_dp = Hash.new, Hash.new
      t1_dp[:x], t2_dp[:x] = [], []

      team1 = match['participants'].take(5)
      t1_dp[:x] << get_team_gold_earned(team1)/match['matchDuration']
      t1_dp[:x] << get_team_kda(team1)
      t1_dp[:x] << get_team_cs_rate(team1)

      team2 = match['participants'].drop(5)
      t2_dp[:x] << get_team_gold_earned(team2)/match['matchDuration']
      t2_dp[:x] << get_team_kda(team2)
      t2_dp[:x] << get_team_cs_rate(team2)

      if match['teams'].first['winner']
        # team 1 is the winner
        t1_dp[:y] = 1
        t2_dp[:y] = 0
      else
        # team 2 is the winner
        t1_dp[:y] = 0
        t2_dp[:y] = 1
      end

      data << t1_dp
      data << t2_dp
    end
    data
  end
end
