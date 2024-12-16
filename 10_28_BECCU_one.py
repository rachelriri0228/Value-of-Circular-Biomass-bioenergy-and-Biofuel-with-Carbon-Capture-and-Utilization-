# -*- coding: utf-8 -*-
"""
Created on Sun Aug 18 14:55:04 2024

@author: Rui Zhou
"""

import numpy as np
from gurobipy import * # Import Gurobi solver
import pandas as pd
import xlrd
import openpyxl

model = Model('Linear Program')

# Set the integrality tolerance
model.Params.OptimalityTol = 0.01  # Adjust the value as need
# Set the optimization gap (e.g., 1%)
optimization_gap = 0.03
model.Params.MIPGap = optimization_gap
# Set the time limit to 600 seconds (10 minutes)
model.setParam('TimeLimit', 1200)

#Index=========================================================================
I = 74 # Set of elevator
L = 291 # Set of biomass farms low 312 mid 295 high 291
J = 88 # Set of potential bioenergy and biofuel production plants
M = 3 # Set of biomasses
T = 12 # Set of planning periods
scenario = 'high'
#Parameter=====================================================================
#read data
#path = '/Users/User/OneDrive - University of Tennessee/Bioenergy/Data/Real data/' #win
#path = '/Users/rachelriri/Library/CloudStorage/OneDrive-UniversityofTennessee/Bioenergy/Data/Test data/' #mac
#path = 'C:/Users/rache/OneDrive - University of Tennessee/Bioenergy/Data/Test data/' #win2
#path = '/Users/ruizhou/Library/CloudStorage/OneDrive-UniversityofTennessee/Bioenergy/Data/Real data/' #mac2
path = '/Users/Rui Zhou/OneDrive - University of Tennessee/Bioenergy/Data/Real data/' #win 303

gamma = 0.1 	#Ratio of CO2 and hydrogen to produce fuels
r = 0.5	    #Conversion rate of CO2 to fuels (%)

biochar_price = 131 #$/ton https://cloverly.com/ultimate-business-guide-to-biochar/#:~:text=Biochar%20price%3A%20The%20average%20biochar,their%20portfolio%20of%20carbon%20credits.
biofuel_price = 1223 #*0.6427 is levelized price #$/ton
diesel_price = 1142 #$/ton
steam_price = 150 * 1.1 #https://www.procurementresource.com/resource-center/steam-price-trends

 #Momonthly degradation rate of biomass m during storage in period t (%)
de_tm = np.array([
    [0.02, 0.01, 0.01],
    [0.02, 0.01, 0.01],
    [0.02, 0.01, 0.01],
    [0.02, 0.01, 0.01],
    [0.02, 0.01, 0.01],
    [0.02, 0.01, 0.01],
    [0.02, 0.01, 0.01],
    [0.02, 0.01, 0.01],
    [0.02, 0.01, 0.01],
    [0.02, 0.01, 0.01],
    [0.02, 0.01, 0.01],
    [0.02, 0.01, 0.01]
])

SA = 1
h= 3000 #h2 price
pc_m = np.array([50,65,70])#Unit purchasing cost of biomass from elevator  
dc_m = np.array([0.2,0.18,0.22]) #unit transportation cost of biomass $/ton-mile

alpha_f_m = np.array([0.675*0.65,0.37*0.65,0.242*0.65]) #Conversion rate of biomass to biofuel
alpha_c_m = np.array([0.17,0.25,0.303]) #Conversion rate of biomass to biochar
alpha_e_m = np.array([0.13,0.13,0.13]) #Conversion rate of biomass to steam
sigma_m = np.array([0.148,0.311,0.133])   #CO2 emission rate of processing biomass to biofuel
        
PC_j = pd.read_excel(path + 'Production capacity of biorefinery.xlsx').iloc[:,0:].values.ravel()/12 #Production capacity of plant /12 to get the monthly capacity


cc_j = pd.read_excel(path + 'Fixed annual cost.xlsx').iloc[:,0:].values.ravel()  #Capital investment of biorefinery
oc_j = pd.read_excel(path + 'Annual operating cost of biorefinery.xlsx').iloc[:,0:].values.ravel() #Annual operating cost of biorefinery
cm_jm = np.asmatrix(pd.read_excel(path +'Additional fixed annual cost of biorefinery for having capability of handling biomass m.xlsx').iloc[:,:].values) #addtional fixed annnual cost for having capability of handling biomass m
sc_i_m = np.asmatrix(pd.read_excel(path +'/Unit storage cost of biomass at elevator.xlsx').iloc[:,0:].values)  #Unit storage cost of biomass at elevator
d_ij = np.asmatrix(pd.read_excel(path +'/Distance between elevator and biorefinery.xlsx').iloc[:,1:].values)

V_ltm = np.zeros((L, 12, 3))
f_name = 'Maximum amount of available biomass at farm'
# Load the workbook
book = openpyxl.load_workbook(path + f_name + '.xlsx')
# Select the sheet by name
sheet = book[scenario]
for l in range(0, L):
    for t in range(0, T):
        for m in range(0, M):
            row = 12 * l + t + 2
            col = m + 3
            V_ltm[l][t][m] = sheet.cell(row=row, column=col).value
# Replace NaN values in V_ltm
V_ltm = np.nan_to_num(V_ltm, nan=0.0)            

#profit rate for converting one ton of biomass m from elevator i at plant j into bioproducts after considering the operating cost, transportation cost, hydrogen cost and revenues from all three bioproducts in month t. 
#r_ijt_m = 

#Decision variables=========================================================== 
#Amount of biomass m from farm l to elevator i in period t (tons/month)
u_lit_m	= {} 
for l in range(0,L):
    for i in range(0,I):
        for t in range(0,T):
            for m in range(0,M):
                var_name = f'u_lit_m_{l}_{i}_{t}_{m}'
                u_lit_m[l,i,t,m] = model.addVar(lb=0.0,ub=GRB.INFINITY,vtype=GRB.CONTINUOUS,name=var_name)

#Amount of biomass m transported from region i to plant j in period t (tons/ month)	
x_ijt_m	= {} 
for i in range(0,I):
    for j in range(0,J):
        for t in range(0,T):
            for m in range(0,M):
                var_name = f'x_ijt_m_{i}_{j}_{t}_{m}'
                x_ijt_m[i,j,t,m] = model.addVar(lb=0.0,ub=GRB.INFINITY,vtype=GRB.CONTINUOUS,name=var_name)

    
#Amount of biomass m stored at elevator= i in period t (tons/ month)
s_it_m = {} 		
for i in range(0,I):
    for t in range(0,T):
        for m in range(0,M):
            var_name = f's_it_m_{i}_{t}_{m}'
            s_it_m[i,t,m] = model.addVar(lb=0.0,ub=GRB.INFINITY,vtype=GRB.CONTINUOUS,name=var_name)   		

#Whether plant j will be selected
z_j = {} 	
for j in range(0,J):
    var_name = f'z_j_{j}'
    z_j[j] = model.addVar(vtype=GRB.BINARY,name=var_name)   

                
y_j_m = {}  # Binary variable indicating if biomass type m is chosen for i -> j at month t
for j in range(0,J):
    for m in range(0,M):
        y_j_m[j, m] = model.addVar(vtype=GRB.BINARY, name=f"y_{j}_{m}")
    
#Objective function==========================================maintains conversion balance at each facility j=================
model.setObjective(quicksum(u_lit_m[l,i,t,m] * pc_m[m] for l in range (0,L) for i in range(0,I) for m in range(0,M) for t in range(T))#biomass purchase cost from elevator
                    + quicksum(sc_i_m[i,m] * s_it_m[i,t,m] for i in range(0,I) for m in range(0,M) for t in range(0,T))#storage
                    + quicksum(cc_j[j] * z_j[j] for j in range(0,J))#capital
                    + quicksum(cm_jm[j,m] * z_j[j] for j in range(0,J) for m in range(M))/3# additional capital
                    + quicksum(oc_j[j] * x_ijt_m[i,j,t,m] for i in range(0,I) for j in range(0,J) for m in range(0,M) for t in range(0,T))#operation                   
                    + quicksum(3.318 * x_ijt_m[i,j,t,m]+ d_ij[i,j] * dc_m[m] * x_ijt_m[i,j,t,m] for i in range(0,I) for j in range(0,J) for m in range(0,M) for t in range(0,T))#transporation
                    + quicksum(x_ijt_m[i,j,t,m] * alpha_f_m[m] * sigma_m[m] * gamma * h for i in range(0,I) for j in range(0,J) for m in range(0,M) for t in range(0,T))#h2
                    - quicksum(x_ijt_m[i,j,t,m] * alpha_f_m[m] * biofuel_price for i in range(0,I) for j in range(0,J) for m in range(0,M) for t in range(0,T))
                    - quicksum(x_ijt_m[i,j,t,m] * alpha_c_m[m] * biochar_price for i in range(0,I) for j in range(0,J) for m in range(0,M) for t in range(0,T))
                    - quicksum(x_ijt_m[i,j,t,m] * alpha_f_m[m] * sigma_m[m] * r * diesel_price  for i in range(0,I) for j in range(0,J) for m in range(0,M) for t in range(0,T))
                    - quicksum(x_ijt_m[i,j,t,m] * alpha_e_m[m] * steam_price for i in range(0,I) for j in range(0,J) for m in range(0,M) for t in range(0,T))
                    ,GRB.MINIMIZE)


# unused_capacity_penalty = 10  # Penalty per unused capacity ton
# model.setObjective(quicksum(u_lit_m[l,i,t,m] * pc_m[m] for l in range (0,L) for i in range(0,I) for m in range(0,M) for t in range(T))#biomass purchase cost from elevator
#                    + quicksum(sc_i_m[i,m] * s_it_m[i,t,m] for i in range(0,I) for m in range(0,M) for t in range(0,T))#storage
#                    + quicksum(cc_j[j] * z_j[j] for j in range(0,J))#capital
#                    + quicksum(cm_jm[j,m] * z_j[j] for j in range(0,J) for m in range(M))/3# additional capital
#                    + quicksum(oc_j[j] * x_ijt_m[i,j,t,m] for i in range(0,I) for j in range(0,J) for m in range(0,M) for t in range(0,T))#operation                   
#                    + quicksum(3.318 * x_ijt_m[i,j,t,m]+ d_ij[i,j] * dc_m[m] * x_ijt_m[i,j,t,m] for i in range(0,I) for j in range(0,J) for m in range(0,M) for t in range(0,T))#transporation
#                    + quicksum(x_ijt_m[i,j,t,m] * alpha_f_m[m] * sigma_m[m] * gamma * h for i in range(0,I) for j in range(0,J) for m in range(0,M) for t in range(0,T))#h2
#                    + unused_capacity_penalty * quicksum(PC_j[j] * (1 - z_j[j]) for j in range(J))
#                    ,GRB.MINIMIZE)




biomass_purchase_cost = model.addVar(lb=0.0,ub=GRB.INFINITY,vtype=GRB.CONTINUOUS,name="biomass_purchase_cost")
biorefinery_fixed_annual_cost = model.addVar(lb=0.0,ub=GRB.INFINITY,vtype=GRB.CONTINUOUS,name="biorefinery_fixed_annual_cost")
biorefinery_addtional_fixed_annual_cost = model.addVar(lb=0.0,ub=GRB.INFINITY,vtype=GRB.CONTINUOUS,name="biorefinery_addtional_fixed_annual_cost")
annual_operating_cost = model.addVar(lb=0.0,ub=GRB.INFINITY,vtype=GRB.CONTINUOUS,name="annual_operating_cost")
transportation_cost  = model.addVar(lb=0.0,ub=GRB.INFINITY,vtype=GRB.CONTINUOUS,name="transportation_cost")
biomass_storage_cost =  model.addVar(lb=0.0,ub=GRB.INFINITY,vtype=GRB.CONTINUOUS,name="biomass_storage_cost")
hydrogen_purchase_cost = model.addVar(lb=0.0,ub=GRB.INFINITY,vtype=GRB.CONTINUOUS,name="hydrogen_purchase_cost")
amount_biomass_farm = model.addVar(lb=0.0,ub=GRB.INFINITY,vtype=GRB.CONTINUOUS,name="amount_biomass_farm")
amount_biomass_elevator = model.addVar(lb=0.0,ub=GRB.INFINITY,vtype=GRB.CONTINUOUS,name="amount_biomass_elevator")
amount_biofuel = model.addVar(lb=0.0,ub=GRB.INFINITY,vtype=GRB.CONTINUOUS,name="amount_biofuel")
amount_biochar = model.addVar(lb=0.0,ub=GRB.INFINITY,vtype=GRB.CONTINUOUS,name="amount_biochar")
amount_diesel = model.addVar(lb=0.0,ub=GRB.INFINITY,vtype=GRB.CONTINUOUS,name="amount_diesel")
amount_steam = model.addVar(lb=0.0,ub=GRB.INFINITY,vtype=GRB.CONTINUOUS,name="amount_steam")
biofuel_revenue = model.addVar(lb=0.0,ub=GRB.INFINITY,vtype=GRB.CONTINUOUS,name="biofuel_revenue")
biochar_revenue = model.addVar(lb=0.0,ub=GRB.INFINITY,vtype=GRB.CONTINUOUS,name="biochar_revenue")
diesel_revenue = model.addVar(lb=0.0,ub=GRB.INFINITY,vtype=GRB.CONTINUOUS,name="diesel_revenue")
steam_revenue = model.addVar(lb=0.0,ub=GRB.INFINITY,vtype=GRB.CONTINUOUS,name="steam_revenue")
amount_captured_CO2 = model.addVar(lb=0.0,ub=GRB.INFINITY,vtype=GRB.CONTINUOUS,name="amount_captured_CO2")

s_i0m = 0 #inital storage
for i in range(0,I):
    for m in range(0,M):
        model.addConstr(s_it_m[i,0,m] == quicksum(u_lit_m[l,i,0,m] for l in range(0,L)) - quicksum(x_ijt_m[i,j,0,m] for j in range(0,J)) + s_i0m*(1-de_tm[0,m]))


for i in range(0,I):
    for m in range(0,M):
        for t in range(1,T):
            model.addConstr(s_it_m[i,t,m] == quicksum(u_lit_m[l,i,t,m] for l in range(0,L)) - quicksum(x_ijt_m[i,j,t,m] for j in range(0,J)) + s_it_m[i,t-1,m]*(1-de_tm[t,m]))

           
for l in range(0,L):
    for m in range(0,M):
        for t in range(0,T):  
            model.addConstr(quicksum(u_lit_m[l,i,t,m] for i in range(0,I)) <= V_ltm[l,t,m],name=f"max avaliable biomass")
            
            
for j in range(0,J):
    for t in range(0,T):
        model.addConstr(quicksum(x_ijt_m[i,j,t,m] * alpha_f_m[m] for i in range(0,I) for m in range(0,M)) <= PC_j[j] * z_j[j],name=f"production capacity ub")
    
# Link the amount of biomass transported to the binary variable
for j in range(0,J):
    for m in range(0,M):
        for i in range(0,T):
            model.addConstr(quicksum(x_ijt_m[i,j,t,m] * alpha_f_m[m] for i in range(0,I)) <= y_j_m[j, m] * PC_j[j], 
                            name=f"link_x_y_{i}_{j}_{m}")
        
# Ensure only one type of biomass is transported
for j in range(0,J):
       model.addConstr(quicksum(y_j_m[j, m] for m in range(0,M)) <= 1* z_j[j], 
                    name=f"one_biomass_{i}_{j}_{t}")          

#==========================================================================================================================================================    
model.addConstr(quicksum(u_lit_m[l,i,t,m] for l in range (0,L) for i in range(0,I) for m in range(0,M) for t in range(0,T))
                        == amount_biomass_farm)
model.addConstr(quicksum(x_ijt_m[i,j,t,m] for i in range(0,I) for j in range(0,J) for m in range(0,M) for t in range(0,T))
                        == amount_biomass_elevator)      
model.addConstr(quicksum(alpha_c_m[m] * x_ijt_m[i,j,t,m] for i in range(0,I) for j in range(0,J) for m in range(0,M) for t in range(0,T))
                        == amount_biochar)
model.addConstr(quicksum(alpha_e_m[m] * x_ijt_m[i,j,t,m] for i in range(0,I) for j in range(0,J) for m in range(0,M) for t in range(0,T))
                        == amount_steam)       
model.addConstr(quicksum(r*sigma_m[m] * alpha_f_m[m] * x_ijt_m[i,j,t,m] for i in range(0,I) for j in range(0,J) for m in range(0,M) for t in range(0,T))
                        == amount_diesel)          
model.addConstr(quicksum(x_ijt_m[i,j,t,m] * alpha_f_m[m] for i in range(0,I) for j in range(0,J) for m in range(0,M) for t in range(0,T))
                        == amount_biofuel)  
 

model.addConstr(quicksum( x_ijt_m[i,j,t,m] * alpha_f_m[m] * biofuel_price for i in range(0,I) for j in range(0,J) for m in range(0,M) for t in range(0,T))
                        == biofuel_revenue) 
model.addConstr(quicksum(alpha_c_m[m] * x_ijt_m[i,j,t,m] * biochar_price for i in range(0,I) for j in range(0,J) for m in range(0,M) for t in range(0,T))
                        == biochar_revenue) 
model.addConstr(quicksum(r*sigma_m[m] * x_ijt_m[i,j,t,m] * diesel_price * alpha_f_m[m] for i in range(0,I) for j in range(0,J) for m in range(0,M) for t in range(0,T))
                        == diesel_revenue)                   
model.addConstr(quicksum(x_ijt_m[i,j,t,m] * alpha_e_m[m] * steam_price for i in range(0,I) for j in range(0,J) for m in range(0,M) for t in range(0,T))
                        == steam_revenue) 

model.addConstr(quicksum(u_lit_m[l,i,t,m] * pc_m[m] for l in range (0,L) for i in range(0,I) for m in range(0,M) for t in range(T))
                    == biomass_purchase_cost)    
model.addConstr(quicksum(cc_j[j] * z_j[j] for j in range(0,J))
                    == biorefinery_fixed_annual_cost)  
model.addConstr(quicksum(cm_jm[j,m] * z_j[j] for j in range(0,J) for m in range(0,M))/3
                    == biorefinery_addtional_fixed_annual_cost)  
model.addConstr(quicksum(oc_j[j] * x_ijt_m[i,j,t,m] for i in range(0,I) for j in range(0,J) for m in range(0,M) for t in range(0,T))
                    == annual_operating_cost)    
model.addConstr(quicksum(3.318 * x_ijt_m[i,j,t,m] + d_ij[i,j] * dc_m[m] * x_ijt_m[i,j,t,m] for i in range(0,I) for j in range(0,J) for m in range(0,M) for t in range(0,T))
                    == transportation_cost)    
model.addConstr(quicksum(sc_i_m[i,m] * s_it_m[i,t,m] for i in range(0,I) for m in range(0,M) for t in range(0,T))
                    == biomass_storage_cost)    
model.addConstr(quicksum(x_ijt_m[i,j,t,m] * alpha_f_m[m] * sigma_m[m] * gamma * h for i in range(0,I) for j in range(0,J) for m in range(0,M) for t in range(0,T))
                    == hydrogen_purchase_cost)  

model.addConstr(quicksum(sigma_m[m] *  x_ijt_m[i,j,t,m] * alpha_f_m[m] for i in range(0,I) for j in range(0,J) for m in range(0,M) for t in range(0,T))
                    == amount_captured_CO2) 


model.write('model.lp') 
model.optimize()     

# Specify the directory path and file name
#directory_path = "/Users/User/OneDrive - University of Tennessee/Bioenergy/Data/Test result/" #win
#directory_path = "/Users/rachelriri/Library/CloudStorage/OneDrive-UniversityofTennessee/Bioenergy/Data/Real result/" #mac
#directory_path = '/Users/ruizhou/Library/CloudStorage/OneDrive-UniversityofTennessee/Bioenergy/Data/Real result/' #mac315
directory_path = '/Users/Rui Zhou/OneDrive - University of Tennessee/Bioenergy/Data/Real result' #win 303
#directory_path = 'C:/Users/rache/OneDrive - University of Tennessee/Bioenergy/Data/Real result/' #win2
  
# Combine the directory path and file name to create the full file path
file_path = os.path.join(directory_path, '10_28_biomass to bioenergy and biofuel with CCS '+scenario+' one.sol')

# Save the model to the specified file path
model.write(file_path)

if model.Status == GRB.TIME_LIMIT:
    print("Optimization stopped due to time limit.")
else:
    print("Optimization completed within the time limit.")


# Print solution==========================================================================================================
print("======objective value =======")
obj = model.getObjective()
print(obj.getValue())

print("====== cost =======")
     
var_name = model.getVarByName("amount_biomass_farm")
print("Biomass_farm:", "%g"%(var_name.X))  

var_name = model.getVarByName("amount_biomass_elevator")
print("Biomass_elevator:", "%g"%(var_name.X))  
        
var_name = model.getVarByName("amount_biofuel")
print("Biofuel:", "%g"%(var_name.X))    
    
var_name = model.getVarByName("amount_biochar")
print("Biochar:", "%g"%(var_name.X))     
    
var_name = model.getVarByName("amount_diesel")
print("Diesel_converted_form_CO2:", "%g"%(var_name.X))     

var_name = model.getVarByName("amount_steam")
print("Steam:", "%g"%(var_name.X)) 

var_name = model.getVarByName("amount_captured_CO2")
print("CO2:", "%g"%(var_name.X)) 

var_name = model.getVarByName("biomass_purchase_cost")
print("Biomass_purchase_cost:", "%g"%(var_name.X))

var_name = model.getVarByName("biorefinery_fixed_annual_cost")
print("Biorefinery_capital_cost:", "%g"%(var_name.X))

var_name = model.getVarByName("annual_operating_cost")
print("Biorefinery_annual_operating_cost:", "%g"%(var_name.X))

var_name = model.getVarByName("biorefinery_addtional_fixed_annual_cost")
print("Biorefinery_addtional_opearting_cost:", "%g"%(var_name.X))

var_name = model.getVarByName("transportation_cost")
print("Total_transportation_cost:", "%g"%(var_name.X))

var_name = model.getVarByName("biomass_storage_cost")
print("Biomass_storage_cost:", "%g"%(var_name.X))

var_name = model.getVarByName("hydrogen_purchase_cost")
print("Hydrogen_purchase_cost:", "%g"%(var_name.X))
 
var_name = model.getVarByName("biofuel_revenue")
print("Biofuel_revenue:", "%g"%(var_name.X))    
    
var_name = model.getVarByName("biochar_revenue")
print("Biochar_revenue:", "%g"%(var_name.X))     
    
var_name = model.getVarByName("diesel_revenue")
print("Diesel_revenue:", "%g"%(var_name.X))         
        
var_name = model.getVarByName("steam_revenue")
print("Steam_revenue:", "%g"%(var_name.X))     


    
if model.Status == GRB.TIME_LIMIT:
    print("Optimization stopped due to time limit.")
else:
    print("Optimization completed within the time limit.")
    
# # Output the total revenue and costs to assess the loss
# print("Total Revenue:", biofuel_revenue.getValue() + biochar_revenue.getValue() + diesel_revenue.getValue() + steam_revenue.getValue())
# print("Total Cost:", model.getObjective().getValue())
# print("Potential Loss:", model.getObjective().getValue() - (biofuel_revenue.getValue() + biochar_revenue.getValue() + diesel_revenue.getValue() + steam_revenue.getValue()))
T    