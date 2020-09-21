# Use linear programming to optimize the salary for fanduel NBA contests.
# Inspired by :
# https://medium.com/ml-everything/using-python-and-linear-programming-to-optimize-fantasy-football-picks-dc9d1229db81
# and ported code from https://github.com/breeko/Fantasy_LP/blob/master/fantasy_lp_final.ipynb from NFL to NBA

# Use data from daily context to optimize salary
# SALARY: 60000
# Required positions: 2 PG's. 2 SG's, 2 SF's, 2 PF's, and 1 C
from pulp import *
import pandas as pd

class SalaryOptimizer():
  def __init__(self,required_positions,salary_cap):
    '''

    :param required_positions: dictionary indicating the constrains on each position
      {
      "PG" : 2
      "SG" : 2
      "SF" : 2
      "PF" : 2
      "C" :  1
      }
    '''
    self.req_pos = required_positions
    self.cap = salary_cap
    self.vars = {}
    self.salaries = {}
    self.points = {}
    self.prob = LpProblem("Fantasy", LpMaximize)

  def __pivot(self,df):
    # pivot salaries and points on position
    for pos in df.position.unique():
      available_pos = df[df.position == pos]
      salary = list(available_pos[["name", "salary"]].set_index("name").to_dict().values())[0]
      point = list(available_pos[["name", "points"]].set_index("name").to_dict().values())[0]
      self.salaries[pos] = salary
      self.points[pos] = point

  def __lp_vars(self):
    # create variables for each position. 0 or 1 to represent if player is included or excluded
    self.vars = {k: LpVariable.dict(k, v, cat="Binary") for k, v in self.points.items()}

  def __solve(self):
    rewards = []
    costs = []
    position_constraints = []

    # Setting up the reward
    for k, v in self.vars.items():
      costs += lpSum([self.salaries[k][i] * self.vars[k][i] for i in v])
      rewards += lpSum([self.points[k][i] * self.vars[k][i] for i in v])
      self.prob += lpSum([self.vars[k][i] for i in v]) <= self.req_pos[k]

    self.prob += lpSum(rewards)
    self.prob += lpSum(costs) <= self.cap
    self.prob.solve()

  def __get_positions(self,pos_names):
    pos = pos_names.split("_")[0]
    return pos

  def __get_names(self,pos_names):
    player = pos_names.split("_")
    player_name = ''.join(player[1:])
    return player_name

  def __get_data_from_pretty(self,pretty):
    data = float(pretty.strip().split("*")[0])
    return data

  def __get_lineup_DF(self):
    div = '---------------------------------------\n'
    #print("Variables:\n")
    names = []
    score = str(self.prob.objective)
    constraints = [str(const) for const in self.prob.constraints.values()]
    for v in self.prob.variables():
      score = score.replace(v.name, str(v.varValue))
      constraints = [const.replace(v.name, str(v.varValue)) for const in constraints]
      if v.varValue != 0:
        #print(v.name, "=", v.varValue)
        names.append(v.name)
    #print(div)
    #print("Constraints:")
    for constraint in constraints:
      constraint_pretty = " + ".join(re.findall("[0-9\.]*\*1.0", constraint))
      #if constraint_pretty != "":
      #  print("{} = {}".format(constraint_pretty, eval(constraint_pretty)))
    #print(div)
    #print("Score:")
    score_pretty = " + ".join(re.findall("[0-9\.]+\*1.0", score))
    #print("{} = {}".format(score_pretty, eval(score)))

    positions = list(map(self.__get_positions(),names))
    player_names = list(map(self.__get_names,names))
    sal = list(map(self.__get_data_from_pretty,constraint_pretty.split('+')))
    points = list(map(self.__get_data_from_pretty,score_pretty.split("+")))
    df = pd.DataFrame({'position':positions,'name':player_names,'points':points,'salary':sal})
    df['pts/$ooo'] = df['points'] / (df['salary'] / 1000)
    return df

  def optimize(self,df):
    '''

    :param df: data frame containing data for 1 day
    'position','name','salary','points'
    :return: a list containing the best lineup along with salary, and score
    '''

    # pivot salaries and points on position
    self.__pivot(df)
    # create variables for each position. 0 or 1 to represent if player is included or excluded
    self.__lp_vars()
    # solve the problem
    self.__solve()
    lineup = self.__get_lineup_DF()

    return lineup


