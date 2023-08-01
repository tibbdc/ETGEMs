"""ETGEMs_function.py

The code in this file reflects the Pyomo Concretemodel construction method of constrainted model. On the basis of this file, with a little modification, you can realize the constraints and object switching of various constrainted models mentioned in our manuscript.

"""

# IMPORTS
# External modules
import cobra
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyomo.environ as pyo
from pyomo.environ import *
from pyomo.opt import SolverFactory


#Extracting information from GEM (iML1515 model)
def Get_Model_Data(model):
    """Returns reaction_list,metabolite_list,lb_list,ub_list,coef_matrix from model.
    
    Notes: 
    ----------
    *model： is in SBML format (.xml).
    """
    reaction_list=[]
    metabolite_list=[]
    lb_list={}
    ub_list={}
    coef_matrix={}
    for rea in model.reactions:
        reaction_list.append(rea.id)
        lb_list[rea.id]=rea.lower_bound
        ub_list[rea.id]=rea.upper_bound
        for met in model.metabolites:
            metabolite_list.append(met.id)
            try:
                rea.get_coefficient(met.id)  
            except:
                pass
            else:
                coef_matrix[met.id,rea.id]=rea.get_coefficient(met.id)
    reaction_list=list(set(reaction_list))
    metabolite_list=list(set(metabolite_list))
    return(reaction_list,metabolite_list,lb_list,ub_list,coef_matrix)

#Encapsulating parameters used in Concretemodel
def Get_Concretemodel_Need_Data(reaction_g0_file,metabolites_lnC_file,model,reaction_kcat_MW_file):
    Concretemodel_Need_Data={}
    reaction_g0=pd.read_csv(reaction_g0_file,index_col=0,sep='\t')
    Concretemodel_Need_Data['reaction_g0']=reaction_g0
    metabolites_lnC = pd.read_csv(metabolites_lnC_file, index_col=0, sep='\t')
    Concretemodel_Need_Data['metabolites_lnC']=metabolites_lnC
    # model=cobra.io.read_sbml_model(model_file)
    #cobra.manipulation.modify.convert_to_irreversible(model)
    reaction_kcat_MW=pd.read_csv(reaction_kcat_MW_file,index_col=0)
    Concretemodel_Need_Data['model']=model
    Concretemodel_Need_Data['reaction_kcat_MW']=reaction_kcat_MW
    [reaction_list,metabolite_list,lb_list,ub_list,coef_matrix]=Get_Model_Data(model)
    Concretemodel_Need_Data['reaction_list']=reaction_list
    Concretemodel_Need_Data['metabolite_list']=metabolite_list
    Concretemodel_Need_Data['lb_list']=lb_list
    Concretemodel_Need_Data['ub_list']=ub_list
    Concretemodel_Need_Data['coef_matrix']=coef_matrix
    return (Concretemodel_Need_Data)


#set_obj_value,set_metabolite,set_Df: only 'True' and 'False'
def Template_Concretemodel(reaction_list=None,metabolite_list=None,coef_matrix=None,metabolites_lnC=None,reaction_g0=None,reaction_kcat_MW=None,lb_list=None,\
    ub_list=None,obj_name=None,K_value=None,obj_target=None,set_obj_value=False,set_substrate_ini=False,substrate_name=None,substrate_value=None,\
    set_product_ini=False,product_value=None,product_id=None,set_metabolite=False,set_Df=False,set_obj_B_value=False,set_stoi_matrix=False,\
    set_bound=False,set_enzyme_constraint=False,set_integer=False,set_metabolite_ratio=False,set_thermodynamics=False,B_value=None,\
    set_obj_E_value=False,set_obj_V_value=False,set_obj_TM_value=False,set_obj_Met_value=False,set_obj_single_E_value=False,E_total=None,\
    Bottleneck_reaction_list=None,set_Bottleneck_reaction=False):
    
    """According to the parameter conditions provided by the user, the specific pyomo model is returned.

    Notes
    ----------
    * reaction_list: List of reaction IDs for the model.
    It is obtained by analyzing SBML model with get_data(model) function.
    
    * metabolite_list: List of metabolite IDs for the model.
    It is obtained by analyzing SBML model with get_data(model) function.
    
    * coef_matrix: The model coefficient matrix.
    It is obtained by analyzing SBML model with get_data(model) function.
    
    * metabolites_lnC:Upper and lower bound information of metabolite concentration (Natural logarithm).
    The format is as follows (in .txt file):
    id	lnClb	lnCub
    2pg_c	-14.50865774	-3.912023005
    13dpg_c	-14.50865774	-3.912023005
    ...
    
    * reaction_g0: Thermodynamic parameter of reactions (drG'°) .
    The format is as follows (in .txt file):
    reaction	g0
    13PPDH2	-21.3
    13PPDH2_reverse	21.3
    ...

    * reaction_kcat_MW: The enzymatic data of kcat (turnover number, /h) divided by MW (molecular weight, kDa).
    The format is as follows (in .csv file):
    kcat,MW,kcat_MW
    AADDGT,49389.2889,40.6396,1215.299582180927
    GALCTND,75632.85923999999,42.5229,1778.63831582512
    ...
    
    * lb_list: Lower bound of reaction flux. It is obtained by analyzing SBML model with get_data(model) function.
    * ub_list: Upper bound of reaction flux. It is obtained by analyzing SBML model with get_data(model) function.
    * obj_name: Name of object, such as set_obj_value, set_obj_single_E_value, set_obj_TM_value and set_obj_Met_value.    
    * K_value: the maximum value minus the minimum value of reaction thermodynamic driving force (1249, maxDFi-minDFi).
    * obj_target: Type of object function (maximize or minimize).
    * set_obj_value: Set the flux as the object function (True or False).
    * set_substrate_ini: Adding substrate amount constraints (True or False).
    * substrate_name: Substrate input reaction ID in the model (such as EX_glc__D_e_reverse).
    * substrate_value: Set the upper bound for substrate input reaction flux (10 mmol/h/gDW).
    * set_product_ini: Set the lower bound for product synthesis reaction flux (True or False).
    * product_value: The lower bound of product synthesis reaction flux.
    * product_id: Product synthesis reaction ID in the model (for biomass: BIOMASS_Ec_iML1515_core_75p37M).
    * set_metabolite: Set the upper and lower bounds of metabolite concentration (True or False).
    * set_Df: Adding thermodynamic driving force expression for reactions (True or False).
    * set_obj_B_value: The object function is the maximizing thermodynamic driving force of a pathway (True or False)
    * set_stoi_matrix: Adding flux balance constraints (True or False).
    * set_bound: Set the upper and lower bounds of reaction flux (True or False).
    * set_enzyme_constraint: Adding enzymamic constraints (True or False).
    * set_integer: Adding binary variables constraints (True or False).
    * set_metabolite_ratio: Adding concentration ratio constraints for metabolites (True or False).
    * set_thermodynamics: Adding thermodynamic constraints (True or False).
    * B_value: The value of maximizing the minimum thermodynamic driving force (MDF).
    * set_obj_E_value: The object function is the minimum enzyme cost of a pathway (True or False).
    * set_obj_V_value: The object function is the pFBA of a pathway (True or False)
    * set_obj_TM_value: The object function is the thermodynamic driving force of a reaction (True or False).
    * set_obj_Met_value: The object function is the concentration of a metabolite (True or False).
    * set_obj_single_E_value: The object function is the enzyme cost of a reaction (True or False).     
    * E_total: Total amount constraint of enzymes (0.13).
    * Bottleneck_reaction_list: A list extracted from the result file automatically.
    * set_Bottleneck_reaction: Adding integer variable constraints for specific reaction (True or False).
    """
    Concretemodel = ConcreteModel()
    Concretemodel.metabolite = pyo.Var(metabolite_list,  within=Reals)
    Concretemodel.Df = pyo.Var(reaction_list,  within=Reals)
    Concretemodel.B = pyo.Var()
    Concretemodel.reaction = pyo.Var(reaction_list,  within=NonNegativeReals)
    Concretemodel.z = pyo.Var(reaction_list,  within=pyo.Binary)
    
    #Set upper and lower bounds of metabolite concentration
    if set_metabolite:
        def set_metabolite(m,i):
            return  inequality(metabolites_lnC.loc[i,'lnClb'], m.metabolite[i], metabolites_lnC.loc[i,'lnCub'])
        Concretemodel.set_metabolite= Constraint(metabolite_list,rule=set_metabolite)        

    #thermodynamic driving force expression for reactions
    if set_Df:
        def set_Df(m,j):
            return  m.Df[j]==-reaction_g0.loc[j,'g0']-2.579*sum(coef_matrix[i,j]*m.metabolite[i]  for i in metabolite_list if (i,j) in coef_matrix.keys())
        Concretemodel.set_Df = Constraint(list(reaction_g0.index),rule=set_Df)
    
    #Set the maximum flux as the object function
    if set_obj_value:   
        def set_obj_value(m):
            return m.reaction[obj_name]
        if obj_target=='maximize':
            Concretemodel.obj = Objective(rule=set_obj_value, sense=maximize)
        elif obj_target=='minimize':
            Concretemodel.obj = Objective(rule=set_obj_value, sense=minimize)
            
    #Set the value of maximizing the minimum thermodynamic driving force as the object function
    if set_obj_B_value:             
        def set_obj_B_value(m):
            return m.B  
        Concretemodel.obj = Objective(rule=set_obj_B_value, sense=maximize)  

    #Set the minimum enzyme cost of a pathway as the object function
    if set_obj_E_value:             
        def set_obj_E_value(m):
            return sum(m.reaction[j]/(reaction_kcat_MW.loc[j,'kcat_MW']) for j in reaction_kcat_MW.index)
        Concretemodel.obj = Objective(rule=set_obj_E_value, sense=minimize)  

    #Minimizing the flux sum of pathway (pFBA)
    if set_obj_V_value:             
        def set_obj_V_value(m):
            return sum(m.reaction[j] for j in reaction_list)
        Concretemodel.obj = Objective(rule=set_obj_V_value, sense=minimize)  

    #To calculate the variability of enzyme usage of single reaction.
    if set_obj_single_E_value:             
        def set_obj_single_E_value(m):
            return m.reaction[obj_name]/(reaction_kcat_MW.loc[obj_name,'kcat_MW'])
        if obj_target=='maximize':
            Concretemodel.obj = Objective(rule=set_obj_single_E_value, sense=maximize) 
        elif obj_target=='minimize':
            Concretemodel.obj = Objective(rule=set_obj_single_E_value, sense=minimize)

    #To calculate the variability of thermodynamic driving force (Dfi) of single reaction.
    if set_obj_TM_value:   
        def set_obj_TM_value(m):
            return m.Df[obj_name]
        if obj_target=='maximize':
            Concretemodel.obj = Objective(rule=set_obj_TM_value, sense=maximize)
        elif obj_target=='minimize':
            Concretemodel.obj = Objective(rule=set_obj_TM_value, sense=minimize)

    #To calculate the concentration variability of metabolites.
    if set_obj_Met_value:   
        def set_obj_Met_value(m):
            return m.metabolite[obj_name]
        if obj_target=='maximize':
            Concretemodel.obj = Objective(rule=set_obj_Met_value, sense=maximize)
        elif obj_target=='minimize':
            Concretemodel.obj = Objective(rule=set_obj_Met_value, sense=minimize)

    #Adding flux balance constraints （FBA）
    if set_stoi_matrix:
        def set_stoi_matrix(m,i):
            return sum(coef_matrix[i,j]*m.reaction[j]  for j in reaction_list if (i,j) in coef_matrix.keys() )==0
        Concretemodel.set_stoi_matrix = Constraint( metabolite_list,rule=set_stoi_matrix)

    #Adding the upper and lower bound constraints of reaction flux
    if set_bound:
        def set_bound(m,j):
            return inequality(lb_list[j],m.reaction[j],ub_list[j])
        Concretemodel.set_bound = Constraint(reaction_list,rule=set_bound) 

    #Set the upper bound for substrate input reaction flux
    if set_substrate_ini:
        def set_substrate_ini(m): 
            return m.reaction[substrate_name] <= substrate_value
        Concretemodel.set_substrate_ini = Constraint(rule=set_substrate_ini)   
        
    #Set the lower bound for product synthesis reaction flux
    if set_product_ini:
        def set_product_ini(m): 
            return m.reaction[product_id] >=product_value
        Concretemodel.set_product_ini = Constraint(rule=set_product_ini)  

    #Adding enzymamic constraints
    if set_enzyme_constraint:
        def set_enzyme_constraint(m):
            return sum( m.reaction[j]/(reaction_kcat_MW.loc[j,'kcat_MW']) for j in reaction_kcat_MW.index)<= E_total
        Concretemodel.set_enzyme_constraint = Constraint(rule=set_enzyme_constraint)

    #Adding thermodynamic MDF(B) object function
    if set_obj_B_value:
        def set_obj_B_value(m,j):
            return m.B<=(m.Df[j]+(1-m.z[j])*K_value)
        Concretemodel.set_obj_B_value = Constraint(reaction_list, rule=set_obj_B_value)

    #Adding thermodynamic constraints
    if set_thermodynamics:
        def set_thermodynamics(m,j):
            return (m.Df[j]+(1-m.z[j])*K_value)>= B_value
        Concretemodel.set_thermodynamics = Constraint(reaction_list, rule=set_thermodynamics)
        
    #Adding binary variables constraints
    if set_integer:
        def set_integer(m,j):
            return m.reaction[j]<=m.z[j]*ub_list[j] 
        Concretemodel.set_integer = Constraint(reaction_list,rule=set_integer)    

    #Adding concentration ratio constraints for metabolites
    if set_metabolite_ratio:
        def set_atp_adp(m):
            return m.metabolite['atp_c']-m.metabolite['adp_c']==np.log(10)
        def set_adp_amp(m):
            return m.metabolite['adp_c']-m.metabolite['amp_c']==np.log(1)
        def set_nad_nadh(m):
            return m.metabolite['nad_c']-m.metabolite['nadh_c']==np.log(10)
        def set_nadph_nadp(m):
            return m.metabolite['nadph_c']-m.metabolite['nadp_c']==np.log(10)
        def set_hco3_co2(m):
            return m.metabolite['hco3_c']-m.metabolite['co2_c']==np.log(2)

        Concretemodel.set_atp_adp = Constraint(rule=set_atp_adp) 
        Concretemodel.set_adp_amp = Constraint(rule=set_adp_amp) 
        Concretemodel.set_nad_nadh = Constraint(rule=set_nad_nadh) 
        Concretemodel.set_nadph_nadp = Constraint(rule=set_nadph_nadp) 
        Concretemodel.set_hco3_co2 = Constraint(rule=set_hco3_co2)

    #Adding Bottleneck reaction constraints
    if set_Bottleneck_reaction:
        def set_Bottleneck_reaction(m,j):
            return m.z[j]==1 
        Concretemodel.set_Bottleneck_reaction = Constraint(Bottleneck_reaction_list,rule=set_Bottleneck_reaction) 

    return Concretemodel

def Get_Max_Min_Df(Concretemodel_Need_Data,obj_name,obj_target,solver):
    reaction_list=Concretemodel_Need_Data['reaction_list']
    metabolite_list=Concretemodel_Need_Data['metabolite_list']
    coef_matrix=Concretemodel_Need_Data['coef_matrix']
    metabolites_lnC=Concretemodel_Need_Data['metabolites_lnC']
    reaction_g0=Concretemodel_Need_Data['reaction_g0']

    Concretemodel = Template_Concretemodel(reaction_list=reaction_list,metabolite_list=metabolite_list,coef_matrix=coef_matrix,\
        metabolites_lnC=metabolites_lnC,reaction_g0=reaction_g0,obj_name=obj_name,obj_target=obj_target,\
        set_obj_TM_value=True,set_metabolite=True,set_Df=True)
    """Calculation both of maximum and minimum thermodynamic driving force for reactions,without metabolite ratio constraints.
    
    Notes:
    ----------
    * reaction_list: List of reaction IDs for the model.
    It is obtained by analyzing SBML model with get_data(model) function.
    
    * metabolite_list: List of metabolite IDs for the model.
    It is obtained by analyzing SBML model with get_data(model) function.

    * coef_matrix: The model coefficient matrix.
    It is obtained by analyzing SBML model with get_data(model) function.
    
    * metabolites_lnC: Upper and lower bound information of metabolite concentration (Natural logarithm).
    The format is as follows (in .txt file):
    id	lnClb	lnCub
    2pg_c	-14.50865774	-3.912023005
    13dpg_c	-14.50865774	-3.912023005
    ...
    
    * reaction_g0: Thermodynamic parameter of reactions (drG'°) .
    The format is as follows (in .txt file):
    reaction	g0
    13PPDH2	-21.3
    13PPDH2_reverse	21.3
    ...
    
    * obj_name: Object reaction ID.
    * obj_target: Type of object function (maximize or minimize).    
    * set_obj_TM_value: set_obj_TM_value: The object function is the thermodynamic driving force of a reaction (True or False).
    * set_metabolite: Set the upper and lower bounds of metabolite concentration (True or False).
    * set_Df: Adding thermodynamic driving force expression for reactions (True or False).
    """ 
    max_min_Df_list=pd.DataFrame()  
    opt=Model_Solve(Concretemodel,solver)

    if obj_target=='maximize':
        max_min_Df_list.loc[obj_name,'max_value']=opt.obj() 
    elif obj_target=='minimize':
        max_min_Df_list.loc[obj_name,'min_value']=opt.obj() 

    return max_min_Df_list

def Get_Max_Min_Df_Ratio(Concretemodel_Need_Data,obj_name,obj_target,solver):
    reaction_list=Concretemodel_Need_Data['reaction_list']
    metabolite_list=Concretemodel_Need_Data['metabolite_list']
    coef_matrix=Concretemodel_Need_Data['coef_matrix']
    metabolites_lnC=Concretemodel_Need_Data['metabolites_lnC']
    reaction_g0=Concretemodel_Need_Data['reaction_g0']

    Concretemodel = Template_Concretemodel(reaction_list=reaction_list,metabolite_list=metabolite_list,coef_matrix=coef_matrix,\
        metabolites_lnC=metabolites_lnC,reaction_g0=reaction_g0,obj_name=obj_name,obj_target=obj_target,\
        set_obj_TM_value=True,set_metabolite=True,set_Df=True,set_metabolite_ratio=True)
    """Calculation both of maximum and minimum thermodynamic driving force for reactions,with metabolite ratio constraints.
    
    Notes:
    ----------
    * reaction_list: List of reaction IDs for the model.
    It is obtained by analyzing SBML model with get_data(model) function.
    
    * metabolite_list: List of metabolite IDs for the model.
    It is obtained by analyzing SBML model with get_data(model) function.

    * coef_matrix: The model coefficient matrix.
    It is obtained by analyzing SBML model with get_data(model) function.
    
    * metabolites_lnC: Upper and lower bound information of metabolite concentration (Natural logarithm).
    The format is as follows (in .txt file):
    id	lnClb	lnCub
    2pg_c	-14.50865774	-3.912023005
    13dpg_c	-14.50865774	-3.912023005
    ...
    
    * reaction_g0: Thermodynamic parameter of reactions (drG'°) .
    The format is as follows (in .txt file):
    reaction	g0
    13PPDH2	-21.3
    13PPDH2_reverse	21.3
    ...
    
    * obj_name: Object reaction ID.
    * obj_target: Type of object function (maximize or minimize).    
    * set_obj_TM_value: set_obj_TM_value: The object function is the thermodynamic driving force of a reaction (True or False).
    * set_metabolite: Set the upper and lower bounds of metabolite concentration (True or False).
    * set_Df: Adding thermodynamic driving force expression for reactions (True or False).
    * set_metabolite_ratio: Adding concentration ratio constraints for metabolites (True or False).   
    """
    max_min_Df_list=pd.DataFrame()
    opt = Model_Solve(Concretemodel,solver)   
    if obj_target=='maximize':
        max_min_Df_list.loc[obj_name,'max_value']=opt.obj() 
    elif obj_target=='minimize':
        max_min_Df_list.loc[obj_name,'min_value']=opt.obj() 

    return max_min_Df_list

#Solving the MDF (B value)
def MDF_Calculation(Concretemodel_Need_Data,product_value,product_id,substrate_name,substrate_value,K_value,E_total,solver):
    reaction_list=Concretemodel_Need_Data['reaction_list']
    metabolite_list=Concretemodel_Need_Data['metabolite_list']
    coef_matrix=Concretemodel_Need_Data['coef_matrix']
    metabolites_lnC=Concretemodel_Need_Data['metabolites_lnC']
    reaction_g0=Concretemodel_Need_Data['reaction_g0']
    reaction_kcat_MW=Concretemodel_Need_Data['reaction_kcat_MW']
    lb_list=Concretemodel_Need_Data['lb_list']
    ub_list=Concretemodel_Need_Data['ub_list']
    ub_list[substrate_name]=substrate_value

    Concretemodel=Template_Concretemodel(reaction_list=reaction_list,metabolite_list=metabolite_list,coef_matrix=coef_matrix,\
        metabolites_lnC=metabolites_lnC,reaction_g0=reaction_g0,reaction_kcat_MW=reaction_kcat_MW,lb_list=lb_list,ub_list=ub_list,\
        K_value=K_value,set_substrate_ini=True,substrate_name=substrate_name,substrate_value=substrate_value,set_product_ini=True,\
        product_value=product_value,product_id=product_id,set_metabolite=True,set_Df=True,set_obj_B_value=True,set_stoi_matrix=True,\
        set_bound=True,set_enzyme_constraint=True,E_total=E_total,set_integer=True,set_metabolite_ratio=True)
    opt=Model_Solve(Concretemodel,solver)
    #B_value=format(Concretemodel.obj(), '.3f')
    B_value=opt.obj()-0.000001
    return B_value

#Constructing a GEM (iML1515 model) using Pyomo Concretemodel framework
def EcoGEM(Concretemodel_Need_Data,obj_name,obj_target,substrate_name,substrate_value):
    reaction_list=Concretemodel_Need_Data['reaction_list']
    metabolite_list=Concretemodel_Need_Data['metabolite_list']
    coef_matrix=Concretemodel_Need_Data['coef_matrix']
    lb_list=Concretemodel_Need_Data['lb_list']
    ub_list=Concretemodel_Need_Data['ub_list']
    ub_list[substrate_name]=substrate_value

    EcoGEM=Template_Concretemodel(reaction_list=reaction_list,metabolite_list=metabolite_list,coef_matrix=coef_matrix,\
            lb_list=lb_list,ub_list=ub_list,obj_name=obj_name,obj_target=obj_target,set_obj_value=True,set_substrate_ini=True,\
            substrate_name=substrate_name,substrate_value=substrate_value,set_stoi_matrix=True,set_bound=True)
    return EcoGEM

#Constructing a enzymatic constraints model (EcoECM) using Pyomo Concretemodel framework
def EcoECM(Concretemodel_Need_Data,obj_name,obj_target,substrate_name,substrate_value,E_total):
    reaction_list=Concretemodel_Need_Data['reaction_list']
    metabolite_list=Concretemodel_Need_Data['metabolite_list']
    coef_matrix=Concretemodel_Need_Data['coef_matrix']
    reaction_kcat_MW=Concretemodel_Need_Data['reaction_kcat_MW']
    lb_list=Concretemodel_Need_Data['lb_list']
    ub_list=Concretemodel_Need_Data['ub_list']
    ub_list[substrate_name]=substrate_value

    EcoECM=Template_Concretemodel(reaction_list=reaction_list,metabolite_list=metabolite_list,coef_matrix=coef_matrix,\
        reaction_kcat_MW=reaction_kcat_MW,lb_list=lb_list,ub_list=ub_list,obj_name=obj_name,obj_target=obj_target,set_obj_value=True,\
        set_substrate_ini=True,substrate_name=substrate_name,substrate_value=substrate_value,set_stoi_matrix=True,set_bound=True,\
        set_enzyme_constraint=True,E_total=E_total)
    return EcoECM

#Constructing a thermodynamic constraints model (EcoTCM) using Pyomo Concretemodel framework
def EcoTCM(Concretemodel_Need_Data,obj_name,obj_target,substrate_name,substrate_value,K_value,B_value):
    reaction_list=Concretemodel_Need_Data['reaction_list']
    metabolite_list=Concretemodel_Need_Data['metabolite_list']
    coef_matrix=Concretemodel_Need_Data['coef_matrix']
    reaction_g0=Concretemodel_Need_Data['reaction_g0']
    metabolites_lnC=Concretemodel_Need_Data['metabolites_lnC']
    lb_list=Concretemodel_Need_Data['lb_list']
    ub_list=Concretemodel_Need_Data['ub_list']
    ub_list[substrate_name]=substrate_value

    EcoTCM=Template_Concretemodel(reaction_list=reaction_list,metabolite_list=metabolite_list,coef_matrix=coef_matrix,\
        metabolites_lnC=metabolites_lnC,reaction_g0=reaction_g0,lb_list=lb_list,ub_list=ub_list,\
        obj_name=obj_name,obj_target=obj_target,set_obj_value=True,set_substrate_ini=True,\
        substrate_name=substrate_name,substrate_value=substrate_value,set_stoi_matrix=True,set_bound=True,set_metabolite=True,\
        set_Df=True,set_integer=True,set_metabolite_ratio=True,set_thermodynamics=True,K_value=K_value,B_value=B_value)
    return EcoTCM

#Constructing a enzymatic and thermodynamic constraints model (EcoETM) using Pyomo Concretemodel framework
def EcoETM(Concretemodel_Need_Data,obj_name,obj_target,substrate_name,substrate_value,E_total,K_value,B_value):
    reaction_list=Concretemodel_Need_Data['reaction_list']
    metabolite_list=Concretemodel_Need_Data['metabolite_list']
    coef_matrix=Concretemodel_Need_Data['coef_matrix']
    reaction_g0=Concretemodel_Need_Data['reaction_g0']
    reaction_kcat_MW=Concretemodel_Need_Data['reaction_kcat_MW']
    metabolites_lnC=Concretemodel_Need_Data['metabolites_lnC']
    lb_list=Concretemodel_Need_Data['lb_list']
    ub_list=Concretemodel_Need_Data['ub_list']
    ub_list[substrate_name]=substrate_value

    EcoETM=Template_Concretemodel(reaction_list=reaction_list,metabolite_list=metabolite_list,coef_matrix=coef_matrix,\
        metabolites_lnC=metabolites_lnC,reaction_g0=reaction_g0,reaction_kcat_MW=reaction_kcat_MW,lb_list=lb_list,ub_list=ub_list,\
        obj_name=obj_name,obj_target=obj_target,set_obj_value=True,set_substrate_ini=True,\
        substrate_name=substrate_name,substrate_value=substrate_value, set_stoi_matrix=True,set_bound=True,set_metabolite=True,\
        set_Df=True,set_integer=True,set_metabolite_ratio=True,set_thermodynamics=True,K_value=K_value,B_value=B_value,\
        set_enzyme_constraint=True,E_total=E_total)
    return EcoETM

#Solving programming problems
def Model_Solve(model,solver):
    opt = pyo.SolverFactory(solver)
    opt.solve(model)
    return model

#Maximum growth rate calculation
def Max_Growth_Rate_Calculation(Concretemodel_Need_Data,obj_name,obj_target,substrate_name,substrate_value,K_value,E_total,B_value,solver):
    reaction_list=Concretemodel_Need_Data['reaction_list']
    metabolite_list=Concretemodel_Need_Data['metabolite_list']
    coef_matrix=Concretemodel_Need_Data['coef_matrix']
    metabolites_lnC=Concretemodel_Need_Data['metabolites_lnC']
    reaction_g0=Concretemodel_Need_Data['reaction_g0']
    reaction_kcat_MW=Concretemodel_Need_Data['reaction_kcat_MW']
    lb_list=Concretemodel_Need_Data['lb_list']
    ub_list=Concretemodel_Need_Data['ub_list']
    ub_list[substrate_name]=substrate_value

    Concretemodel=Template_Concretemodel(reaction_list=reaction_list,metabolite_list=metabolite_list,coef_matrix=coef_matrix,\
        metabolites_lnC=metabolites_lnC,reaction_g0=reaction_g0,reaction_kcat_MW=reaction_kcat_MW,lb_list=lb_list,ub_list=ub_list,\
        obj_name=obj_name,K_value=K_value,obj_target=obj_target,set_obj_value=True,set_substrate_ini=True,\
        substrate_name=substrate_name,substrate_value=substrate_value,set_metabolite=True,set_Df=True,set_stoi_matrix=True,\
        set_bound=True,set_enzyme_constraint=True,E_total=E_total,set_integer=True,set_metabolite_ratio=True,\
        set_thermodynamics=True,B_value=B_value)
    opt=Model_Solve(Concretemodel,solver)
    return opt.obj()

#Minimum enzyme cost calculation
def Min_Enzyme_Cost_Calculation(Concretemodel_Need_Data,product_value,product_id,substrate_name,substrate_value,K_value,E_total,B_value,solver):
    reaction_list=Concretemodel_Need_Data['reaction_list']
    metabolite_list=Concretemodel_Need_Data['metabolite_list']
    coef_matrix=Concretemodel_Need_Data['coef_matrix']
    metabolites_lnC=Concretemodel_Need_Data['metabolites_lnC']
    reaction_g0=Concretemodel_Need_Data['reaction_g0']
    reaction_kcat_MW=Concretemodel_Need_Data['reaction_kcat_MW']
    lb_list=Concretemodel_Need_Data['lb_list']
    ub_list=Concretemodel_Need_Data['ub_list']
    ub_list[substrate_name]=substrate_value

    Concretemodel=Template_Concretemodel(reaction_list=reaction_list,metabolite_list=metabolite_list,coef_matrix=coef_matrix,\
        metabolites_lnC=metabolites_lnC,reaction_g0=reaction_g0,reaction_kcat_MW=reaction_kcat_MW,lb_list=lb_list,ub_list=ub_list,\
        K_value=K_value,set_product_ini=True,product_value=product_value,product_id=product_id,set_substrate_ini=True,\
        substrate_name=substrate_name,substrate_value=substrate_value,set_metabolite=True,set_Df=True,set_stoi_matrix=True,\
        set_bound=True,set_enzyme_constraint=True,E_total=E_total,set_integer=True,set_metabolite_ratio=True,\
        set_thermodynamics=True,B_value=B_value,set_obj_E_value=True)
    opt=Model_Solve(Concretemodel,solver)
    min_E=opt.obj()
    return min_E

#Minimum flux sum calculation（pFBA）
def Min_Flux_Sum_Calculation(Concretemodel_Need_Data,product_value,product_id,substrate_name,substrate_value,K_value,E_total,B_value,solver):
    reaction_list=Concretemodel_Need_Data['reaction_list']
    metabolite_list=Concretemodel_Need_Data['metabolite_list']
    coef_matrix=Concretemodel_Need_Data['coef_matrix']
    metabolites_lnC=Concretemodel_Need_Data['metabolites_lnC']
    reaction_g0=Concretemodel_Need_Data['reaction_g0']
    reaction_kcat_MW=Concretemodel_Need_Data['reaction_kcat_MW']
    lb_list=Concretemodel_Need_Data['lb_list']
    ub_list=Concretemodel_Need_Data['ub_list']
    ub_list[substrate_name]=substrate_value

    Concretemodel=Template_Concretemodel(reaction_list=reaction_list,metabolite_list=metabolite_list,coef_matrix=coef_matrix,\
        metabolites_lnC=metabolites_lnC,reaction_g0=reaction_g0,reaction_kcat_MW=reaction_kcat_MW,lb_list=lb_list,ub_list=ub_list,\
        K_value=K_value,set_product_ini=True,product_value=product_value,product_id=product_id,set_substrate_ini=True,\
        substrate_name=substrate_name,substrate_value=substrate_value,set_metabolite=True,set_Df=True,set_stoi_matrix=True,\
        set_bound=True,set_enzyme_constraint=True,E_total=E_total,set_integer=True,set_metabolite_ratio=True,\
        set_thermodynamics=True,B_value=B_value,set_obj_V_value=True)
    opt=Model_Solve(Concretemodel,solver)

    min_V=opt.obj()
    return [min_V,Concretemodel]

#Determination of bottleneck reactions by analysing the variability of thermodynamic driving force
def Get_Max_Min_Df_Complete(Concretemodel_Need_Data,obj_name,obj_target,K_value,B_value,max_product_under_mdf,product_id,E_total,substrate_name,substrate_value,solver):
    reaction_list=Concretemodel_Need_Data['reaction_list']
    metabolite_list=Concretemodel_Need_Data['metabolite_list']
    coef_matrix=Concretemodel_Need_Data['coef_matrix']
    metabolites_lnC=Concretemodel_Need_Data['metabolites_lnC']
    reaction_g0=Concretemodel_Need_Data['reaction_g0']
    reaction_kcat_MW=Concretemodel_Need_Data['reaction_kcat_MW']
    lb_list=Concretemodel_Need_Data['lb_list']
    ub_list=Concretemodel_Need_Data['ub_list']
    ub_list[substrate_name]=substrate_value

    Concretemodel = Template_Concretemodel(reaction_list=reaction_list,metabolite_list=metabolite_list,coef_matrix=coef_matrix,\
        metabolites_lnC=metabolites_lnC,reaction_g0=reaction_g0,reaction_kcat_MW=reaction_kcat_MW,lb_list=lb_list,\
        ub_list=ub_list,obj_name=obj_name,obj_target=obj_target,set_obj_TM_value=True,set_metabolite=True,set_Df=True,set_metabolite_ratio=True,set_bound=True,\
        set_stoi_matrix=True,set_substrate_ini=True,K_value=K_value,substrate_name=substrate_name,substrate_value=substrate_value,set_thermodynamics=True,B_value=B_value,\
        set_product_ini=True,product_id=product_id,product_value=max_product_under_mdf,set_integer=True,set_enzyme_constraint=True,E_total=E_total)
    """Calculation both of maximum and minimum thermodynamic driving force for reactions in a special list.
    
    Notes:
    ----------
    * reaction_list: List of reaction IDs for the model.
    It is obtained by analyzing SBML model with get_data(model) function.
    
    * metabolite_list: List of metabolite IDs for the model.
    It is obtained by analyzing SBML model with get_data(model) function.

    * coef_matrix: The model coefficient matrix.
    It is obtained by analyzing SBML model with get_data(model) function.
    
    * metabolites_lnC: Upper and lower bound information of metabolite concentration (Natural logarithm).
    The format is as follows (in .txt file):
    id	lnClb	lnCub
    2pg_c	-14.50865774	-3.912023005
    13dpg_c	-14.50865774	-3.912023005
    ...
    
    * reaction_g0: Thermodynamic parameter of reactions (drG'°) .
    The format is as follows (in .txt file):
    reaction	g0
    13PPDH2	-21.3
    13PPDH2_reverse	21.3
    ...
    
    * reaction_kcat_MW: The enzymatic data of kcat (turnover number, /h) divided by MW (molecular weight, kDa).
    The format is as follows (in .csv file):
    kcat,MW,kcat_MW
    AADDGT,49389.2889,40.6396,1215.299582180927
    GALCTND,75632.85923999999,42.5229,1778.63831582512
    ...
    
    * lb_list: Lower bound of reaction flux. It is obtained by analyzing SBML model with get_data(model) function.
    * ub_list: Upper bound of reaction flux. It is obtained by analyzing SBML model with get_data(model) function.
    * ub_list[substrate_name]: substrate_value (the upper bound for substrate input reaction flux)
    * obj_name: Object reaction ID.
    * obj_target: Type of object function (maximize or minimize). 
    * set_obj_TM_value: The object function is the thermodynamic driving force of a reaction (True or False).
    * set_metabolite: Set the upper and lower bounds of metabolite concentration (True or False).
    * set_Df: Adding thermodynamic driving force expression for reactions (True or False).
    * set_metabolite_ratio: Adding concentration ratio constraints for metabolites (True or False).
    * set_bound: Set the upper and lower bounds of reaction flux (True or False).
    * set_stoi_matrix: Adding flux balance constraints (True or False).
    * set_product_ini: Set the lower bound for product synthesis reaction flux (True or False).
    * K_value: the maximum value minus the minimum value of reaction thermodynamic driving force (1249, maxDFi-minDFi).
    * substrate_name: Substrate input reaction ID in the model (such as EX_glc__D_e_reverse).
    * substrate_value: Set the upper bound for substrate input reaction flux (10 mM).
    * set_thermodynamics: Adding thermodynamic constraints (True or False).
    * B_value: The value of maximizing the minimum thermodynamic driving force (MDF).
    * set_product_ini: Set the lower bound for product synthesis reaction flux (True or False).
    * product_id: Product synthesis reaction ID in the model (for biomass: BIOMASS_Ec_iML1515_core_75p37M).
    * product_value: The lower bound of product synthesis reaction flux.
    * set_integer: Adding binary variables constraints (True or False).
    * set_enzyme_constraint: Adding enzymamic constraints (True or False).
    * E_total: Total amount constraint of enzymes (0.13).
    """
    max_min_Df_list=pd.DataFrame()
    opt=Model_Solve(Concretemodel,solver)    
    if obj_target=='maximize':
        max_min_Df_list.loc[obj_name,'max_value']=opt.obj() 
    elif obj_target=='minimize':
        max_min_Df_list.loc[obj_name,'min_value']=opt.obj() 

    return max_min_Df_list

#Determination of limiting metabolites by analysing the concentration variability
def Get_Max_Min_Met_Concentration(Concretemodel_Need_Data,obj_name,obj_target,K_value,B_value,max_product_under_mdf,product_id,E_total,substrate_name,substrate_value,Bottleneck_reaction_list,solver):
    reaction_list=Concretemodel_Need_Data['reaction_list']
    metabolite_list=Concretemodel_Need_Data['metabolite_list']
    coef_matrix=Concretemodel_Need_Data['coef_matrix']
    metabolites_lnC=Concretemodel_Need_Data['metabolites_lnC']
    reaction_g0=Concretemodel_Need_Data['reaction_g0']
    reaction_kcat_MW=Concretemodel_Need_Data['reaction_kcat_MW']
    lb_list=Concretemodel_Need_Data['lb_list']
    ub_list=Concretemodel_Need_Data['ub_list']
    ub_list[substrate_name]=substrate_value

    Concretemodel = Template_Concretemodel(reaction_list=reaction_list,metabolite_list=metabolite_list,coef_matrix=coef_matrix,\
        metabolites_lnC=metabolites_lnC,reaction_g0=reaction_g0,reaction_kcat_MW=reaction_kcat_MW,lb_list=lb_list,\
        ub_list=ub_list,obj_name=obj_name,obj_target=obj_target,set_obj_Met_value=True,set_metabolite=True,set_Df=True,set_metabolite_ratio=True,set_bound=True,\
        set_stoi_matrix=True,set_substrate_ini=True,K_value=K_value,substrate_name=substrate_name,substrate_value=substrate_value,set_thermodynamics=True,B_value=B_value,\
        set_product_ini=True,product_id=product_id,product_value=max_product_under_mdf,set_integer=True,set_enzyme_constraint=True,E_total=E_total,\
        Bottleneck_reaction_list=Bottleneck_reaction_list,set_Bottleneck_reaction=True)
    """Calculation of the maximum and minimum concentrations for metabolites in a specific list.

    Notes：
    ----------
    * reaction_list: List of reaction IDs for the model.
    It is obtained by analyzing SBML model with get_data(model) function.
    
    * metabolite_list: List of metabolite IDs for the model.
    It is obtained by analyzing SBML model with get_data(model) function.

    * coef_matrix: The model coefficient matrix.
    It is obtained by analyzing SBML model with get_data(model) function.
    
    * metabolites_lnC: Upper and lower bound information of metabolite concentration (Natural logarithm).
    The format is as follows (in .txt file):
    id	lnClb	lnCub
    2pg_c	-14.50865774	-3.912023005
    13dpg_c	-14.50865774	-3.912023005
    ...
    
    * reaction_g0: Thermodynamic parameter of reactions (drG'°) .
    The format is as follows (in .txt file):
    reaction	g0
    13PPDH2	-21.3
    13PPDH2_reverse	21.3
    ...
    
    * reaction_kcat_MW: The enzymatic data of kcat (turnover number, /h) divided by MW (molecular weight, kDa).
    The format is as follows (in .csv file):
    kcat,MW,kcat_MW
    AADDGT,49389.2889,40.6396,1215.299582180927
    GALCTND,75632.85923999999,42.5229,1778.63831582512
    ...
    
    * lb_list: Lower bound of reaction flux. It is obtained by analyzing SBML model with get_data(model) function.
    * ub_list: Upper bound of reaction flux. It is obtained by analyzing SBML model with get_data(model) function.
    * obj_name: Object reaction ID.
    * obj_target: Type of object function (maximize or minimize).   
    * set_obj_Met_value: The object function is the concentration of a metabolite (True or False).
    * set_metabolite: Set the upper and lower bounds of metabolite concentration (True or False).
    * set_Df: Adding thermodynamic driving force expression for reactions (True or False).
    * set_metabolite_ratio: Adding concentration ratio constraints for metabolites (True or False).
    * set_bound: Set the upper and lower bounds of reaction flux (True or False).
    * set_stoi_matrix: Adding flux balance constraints (True or False).
    * set_substrate_ini: Adding substrate amount constraints (True or False).
    * K_value: the maximum value minus the minimum value of reaction thermodynamic driving force (1249, maxDFi-minDFi).
    * substrate_name: Substrate input reaction ID in the model (such as EX_glc__D_e_reverse).
    * substrate_value: Set the upper bound for substrate input reaction flux (10 mM).
    * set_thermodynamics: Adding thermodynamic constraints (True or False).
    * B_value: The value of maximizing the minimum thermodynamic driving force (MDF).
    * set_product_ini: Set the lower bound for product synthesis reaction flux (True or False).
    * product_id: Product synthesis reaction ID in the model (for biomass: BIOMASS_Ec_iML1515_core_75p37M).
    * product_value: The lower bound of product synthesis reaction flux.
    * set_enzyme_constraint: Adding enzymamic constraints (True or False).
    * set_integer: Adding binary variables constraints (True or False)
    * E_total: Total amount constraint of enzymes (0.13).
    * Bottleneck_reaction_list: A list extracted from the result file automatically.
    * set_Bottleneck_reaction: Adding integer variable constraints for specific reaction (True or False).
    """  
    max_min_Df_list=pd.DataFrame()
    opt=Model_Solve(Concretemodel,solver)   
    if obj_target=='maximize':
        max_min_Df_list.loc[obj_name,'max_value']=opt.obj() 
    elif obj_target=='minimize':
        max_min_Df_list.loc[obj_name,'min_value']=opt.obj() 

    return max_min_Df_list

#Determination of key enzymes by analysing the enzyme cost variability
def Get_Max_Min_E(Concretemodel_Need_Data,obj_name,obj_target,K_value,B_value,max_product_under_mdf,product_id,E_total,substrate_name,substrate_value,solver):
    reaction_list=Concretemodel_Need_Data['reaction_list']
    metabolite_list=Concretemodel_Need_Data['metabolite_list']
    coef_matrix=Concretemodel_Need_Data['coef_matrix']
    metabolites_lnC=Concretemodel_Need_Data['metabolites_lnC']
    reaction_g0=Concretemodel_Need_Data['reaction_g0']
    reaction_kcat_MW=Concretemodel_Need_Data['reaction_kcat_MW']
    lb_list=Concretemodel_Need_Data['lb_list']
    ub_list=Concretemodel_Need_Data['ub_list']
    ub_list[substrate_name]=substrate_value

    Concretemodel = Template_Concretemodel(reaction_list=reaction_list,metabolite_list=metabolite_list,coef_matrix=coef_matrix,\
        metabolites_lnC=metabolites_lnC,reaction_g0=reaction_g0,reaction_kcat_MW=reaction_kcat_MW,lb_list=lb_list,\
        ub_list=ub_list,obj_name=obj_name,obj_target=obj_target,set_obj_single_E_value=True,set_metabolite=True,set_Df=True,set_metabolite_ratio=True,set_bound=True,\
        set_stoi_matrix=True,set_substrate_ini=True,K_value=K_value,substrate_name=substrate_name,substrate_value=substrate_value,set_thermodynamics=True,B_value=B_value,\
        set_product_ini=True,product_id=product_id,product_value=max_product_under_mdf,set_integer=True,set_enzyme_constraint=True,E_total=E_total)
    """Calculation of the maximum and minimum enzyme cost for reactions in a specific list.

    Notes:
    ----------
    * reaction_list: List of reaction IDs for the model.
    It is obtained by analyzing SBML model with get_data(model) function.
    
    * metabolite_list: List of metabolite IDs for the model.
    It is obtained by analyzing SBML model with get_data(model) function.

    * coef_matrix: The model coefficient matrix.
    It is obtained by analyzing SBML model with get_data(model) function.
    
    * metabolites_lnC: Upper and lower bound information of metabolite concentration (Natural logarithm).
    The format is as follows (in .txt file):
    id	lnClb	lnCub
    2pg_c	-14.50865774	-3.912023005
    13dpg_c	-14.50865774	-3.912023005
    ...
    
    * reaction_g0: Thermodynamic parameter of reactions (drG'°) .
    The format is as follows (in .txt file):
    reaction	g0
    13PPDH2	-21.3
    13PPDH2_reverse	21.3
    ...
    
    * reaction_kcat_MW: The enzymatic data of kcat (turnover number, /h) divided by MW (molecular weight, kDa).
    The format is as follows (in .csv file):
    kcat,MW,kcat_MW
    AADDGT,49389.2889,40.6396,1215.299582180927
    GALCTND,75632.85923999999,42.5229,1778.63831582512
    ...
    
    * lb_list: Lower bound of reaction flux. It is obtained by analyzing SBML model with get_data(model) function.
    * ub_list: Upper bound of reaction flux. It is obtained by analyzing SBML model with get_data(model) function.
    * obj_name: Object reaction ID.
    * obj_target: Type of object function (maximize or minimize).  
    * set_obj_single_E_value: The object function is the enzyme cost of a reaction (True or False).
    * set_metabolite: Set the upper and lower bounds of metabolite concentration (True or False).
    * set_Df: Adding thermodynamic driving force expression for reactions (True or False).
    * set_metabolite_ratio: Adding concentration ratio constraints for metabolites (True or False).
    * set_bound: Set the upper and lower bounds of reaction flux (True or False).
    * set_stoi_matrix: Adding flux balance constraints (True or False).
    * set_substrate_ini: Adding substrate amount constraints (True or False).
    * K_value: the maximum value minus the minimum value of reaction thermodynamic driving force (1249, maxDFi-minDFi).
    * substrate_name: Substrate input reaction ID in the model (such as EX_glc__D_e_reverse).
    * substrate_value: Set the upper bound for substrate input reaction flux (10 mM).
    * set_thermodynamics: Adding thermodynamic constraints (True or False).
    * B_value: The value of maximizing the minimum thermodynamic driving force (MDF).
    * set_product_ini: Set the lower bound for product synthesis reaction flux (True or False).
    * product_id: Product synthesis reaction ID in the model (for biomass: BIOMASS_Ec_iML1515_core_75p37M).
    * product_value: The lower bound of product synthesis reaction flux.
    * set_integer: Adding binary variables constraints (True or False).
    * set_enzyme_constraint: Adding enzymamic constraints (True or False).
    * E_total: Total amount constraint of enzymes (0.13).
    """  

    max_min_Df_list=pd.DataFrame()
    opt=Model_Solve(Concretemodel,solver)
    if obj_target=='maximize':
        max_min_Df_list.loc[obj_name,'max_value']=opt.obj() 
    elif obj_target=='minimize':
        max_min_Df_list.loc[obj_name,'min_value']=opt.obj() 

    return max_min_Df_list

#Solving maximum growth by different models
def Max_OBJ_By_Four_Model(Concretemodel_Need_Data,obj_name,obj_target,substrate_name,substrate_value,K_value,B_value,E_total,solver):
    reaction_list=Concretemodel_Need_Data['reaction_list']
    metabolite_list=Concretemodel_Need_Data['metabolite_list']
    coef_matrix=Concretemodel_Need_Data['coef_matrix']
    metabolites_lnC=Concretemodel_Need_Data['metabolites_lnC']
    reaction_g0=Concretemodel_Need_Data['reaction_g0']
    reaction_kcat_MW=Concretemodel_Need_Data['reaction_kcat_MW']
    lb_list=Concretemodel_Need_Data['lb_list']
    ub_list=Concretemodel_Need_Data['ub_list']
    ub_list[substrate_name]=substrate_value

    product_list=pd.DataFrame()
    ub_list[substrate_name]=substrate_value
    EcoGEM=Template_Concretemodel(reaction_list=reaction_list,metabolite_list=metabolite_list,coef_matrix=coef_matrix,\
        lb_list=lb_list,ub_list=ub_list,obj_name=obj_name,obj_target=obj_target,set_obj_value=True,set_substrate_ini=True,\
        substrate_name=substrate_name,substrate_value=substrate_value,set_stoi_matrix=True,set_bound=True)
    opt=Model_Solve(EcoGEM,solver)
    product_list.loc[substrate_value,'iML1515']=opt.obj()

    EcoECM=Template_Concretemodel(reaction_list=reaction_list,metabolite_list=metabolite_list,coef_matrix=coef_matrix,\
        reaction_kcat_MW=reaction_kcat_MW,lb_list=lb_list,ub_list=ub_list,obj_name=obj_name,obj_target=obj_target,set_obj_value=True,\
        set_substrate_ini=True,substrate_name=substrate_name,substrate_value=substrate_value,set_stoi_matrix=True,set_bound=True,\
        set_enzyme_constraint=True,E_total=E_total)
    opt=Model_Solve(EcoECM,solver)
    product_list.loc[substrate_value,'EcoECM']=opt.obj()

    EcoTCM=Template_Concretemodel(reaction_list=reaction_list,metabolite_list=metabolite_list,coef_matrix=coef_matrix,\
        metabolites_lnC=metabolites_lnC,reaction_g0=reaction_g0,lb_list=lb_list,ub_list=ub_list,\
        obj_name=obj_name,obj_target=obj_target,set_obj_value=True,set_substrate_ini=True,\
        substrate_name=substrate_name,substrate_value=substrate_value,set_stoi_matrix=True,set_bound=True,set_metabolite=True,\
        set_Df=True,set_integer=True,set_metabolite_ratio=True,set_thermodynamics=True,K_value=K_value,B_value=B_value)
    opt=Model_Solve(EcoTCM,solver)
    product_list.loc[substrate_value,'EcoTCM(Dfi>=0)']=opt.obj()

    EcoETM=Template_Concretemodel(reaction_list=reaction_list,metabolite_list=metabolite_list,coef_matrix=coef_matrix,\
        metabolites_lnC=metabolites_lnC,reaction_g0=reaction_g0,reaction_kcat_MW=reaction_kcat_MW,lb_list=lb_list,ub_list=ub_list,\
        obj_name=obj_name,obj_target=obj_target,set_obj_value=True,set_substrate_ini=True,\
        substrate_name=substrate_name,substrate_value=substrate_value, set_stoi_matrix=True,set_bound=True,set_metabolite=True,\
        set_Df=True,set_integer=True,set_metabolite_ratio=True,set_thermodynamics=True,K_value=K_value,B_value=B_value,\
        set_enzyme_constraint=True,E_total=E_total)
    opt=Model_Solve(EcoETM,solver)
    product_list.loc[substrate_value,'EcoETM']=opt.obj()

    return product_list

#Solving MDF value under preset growth rate
def Max_MDF_By_model(Concretemodel_Need_Data,substrate_name,substrate_value,product_value,product_id,K_value,E_total,obj_enz_constraint,obj_no_enz_constraint,solver):
    reaction_list=Concretemodel_Need_Data['reaction_list']
    metabolite_list=Concretemodel_Need_Data['metabolite_list']
    coef_matrix=Concretemodel_Need_Data['coef_matrix']
    metabolites_lnC=Concretemodel_Need_Data['metabolites_lnC']
    reaction_g0=Concretemodel_Need_Data['reaction_g0']
    reaction_kcat_MW=Concretemodel_Need_Data['reaction_kcat_MW']
    lb_list=Concretemodel_Need_Data['lb_list']
    ub_list=Concretemodel_Need_Data['ub_list']
    ub_list[substrate_name]=substrate_value
    MDF_list=pd.DataFrame()

    if product_value<=obj_no_enz_constraint:
        EcoTCM=Template_Concretemodel(reaction_list=reaction_list,metabolite_list=metabolite_list,coef_matrix=coef_matrix,\
            metabolites_lnC=metabolites_lnC,reaction_g0=reaction_g0,lb_list=lb_list,ub_list=ub_list,\
            K_value=K_value,set_substrate_ini=True,substrate_name=substrate_name,substrate_value=substrate_value,set_product_ini=True,\
            product_value=product_value,product_id=product_id,set_metabolite=True,set_Df=True,set_obj_B_value=True,set_stoi_matrix=True,\
            set_bound=True,set_integer=True,set_metabolite_ratio=True)

        opt=Model_Solve(EcoTCM,solver)
        MDF_list.loc[product_value,'EcoTCM']=opt.obj()
    else:
        MDF_list.loc[product_value,'EcoTCM']=None

    if product_value<=obj_enz_constraint:
        EcoETM=Template_Concretemodel(reaction_list=reaction_list,metabolite_list=metabolite_list,coef_matrix=coef_matrix,\
            metabolites_lnC=metabolites_lnC,reaction_g0=reaction_g0,reaction_kcat_MW=reaction_kcat_MW,lb_list=lb_list,ub_list=ub_list,\
            K_value=K_value,set_substrate_ini=True,substrate_name=substrate_name,substrate_value=substrate_value,set_product_ini=True,\
            product_value=product_value,product_id=product_id,set_metabolite=True,set_Df=True,set_obj_B_value=True,set_stoi_matrix=True,\
            set_bound=True,set_enzyme_constraint=True,E_total=E_total,set_integer=True,set_metabolite_ratio=True)

        opt=Model_Solve(EcoETM,solver)
        MDF_list.loc[product_value,'EcoETM']=opt.obj()
    else:
        MDF_list.loc[product_value,'EcoETM']=None
        
    return MDF_list

def Get_Results_Thermodynamics(model,Concretemodel,reaction_kcat_MW,reaction_g0,coef_matrix,metabolite_list):
    """The formatting of the calculated results, includes the metabolic flux, binary variable values, thermodynamic driving force of reactions, the enzyme amount and the metabolite concentrations. The value of "-9999" means that the missing of kinetic (kcat) or thermodynamickcat (drG'°) parameters.
    
    Notes:
    ----------
    * model: is in SBML format (.xml).
    * Concretemodel: Pyomo Concretemodel.
    
    * reaction_kcat_MW: The enzymatic data of kcat (turnover number, /h) divided by MW (molecular weight, kDa).
    The format is as follows (in .csv file):
    kcat,MW,kcat_MW
    AADDGT,49389.2889,40.6396,1215.299582180927
    GALCTND,75632.85923999999,42.5229,1778.63831582512
    ...
    
    * reaction_g0: Thermodynamic parameter of reactions (drG'°) .
    The format is as follows (in .txt file):
    reaction	g0
    13PPDH2	-21.3
    13PPDH2_reverse	21.3
    ...
    
    * coef_matrix: The model coefficient matrix.
    It is obtained by analyzing SBML model with get_data(model) function.
    
    * metabolite_list: List of metabolite IDs for the model.
    It is obtained by analyzing SBML model with get_data(model) function.
    """
    result_dataframe = pd.DataFrame()
    for eachreaction in Concretemodel.reaction:
        flux=Concretemodel.reaction[eachreaction].value
        z=Concretemodel.z[eachreaction].value
        result_dataframe.loc[eachreaction,'flux']=flux
        result_dataframe.loc[eachreaction,'z']=z  
        if eachreaction in reaction_g0.index:
            result_dataframe.loc[eachreaction,'f']=-reaction_g0.loc[eachreaction,'g0']-2.579*sum(coef_matrix[i,eachreaction]*Concretemodel.metabolite[i].value  for i in metabolite_list if (i,eachreaction) in coef_matrix.keys())
        else:
            result_dataframe.loc[eachreaction,'f']=-9999
        if eachreaction in reaction_kcat_MW.index:
            result_dataframe.loc[eachreaction,'enz']= flux/(reaction_kcat_MW.loc[eachreaction,'kcat_MW'])
        else:
            result_dataframe.loc[eachreaction,'enz']= -9999 
            
        tmp=model.reactions.get_by_id(eachreaction)
        met_list=''
        for met in tmp.metabolites:    
            met_list=met_list+';'+str(met.id)+' : '+str(np.exp(Concretemodel.metabolite[met.id].value))
        result_dataframe.loc[eachreaction,'met_concentration']= met_list  
        
    return(result_dataframe)

#Visualization of calculation results
def Draw_product_By_Glucose_rate(product_list):
    plt.figure(figsize=(15, 10), dpi=300)

    plt.plot(product_list.index, product_list[product_list.columns[0]], color="black", linewidth=3.0, linestyle="--", label=product_list.columns[0])
    plt.plot(product_list.index, product_list[product_list.columns[1]], color="red", linewidth=3.0, linestyle="-", label=product_list.columns[1])
    plt.plot(product_list.index, product_list[product_list.columns[2]], color="cyan", linewidth=3.0, linestyle="-", label=product_list.columns[2])
    plt.plot(product_list.index, product_list[product_list.columns[3]], color="darkorange", linewidth=3.0, linestyle="-", label=product_list.columns[3])

    font1 = {'family' : 'Times New Roman',
    'weight' : 'normal',
    'size'   : 20,
    }

    plt.legend(loc="upper left",prop=font1)

    plt.xlim(0, 15)
    plt.ylim(0, 1.4)

    plt.tick_params(labelsize=23)
    plt.xticks([0, 1, 2,3, 4,5, 6, 7, 8, 9, 10, 11, 12, 13,14,15])
    plt.yticks([0.2, 0.4, 0.6,0.8, 1.0,1.2, 1.4])

    font2 = {'family' : 'Times New Roman',
    'weight' : 'normal',
    'size'   : 25,
    }
    plt.xlabel("Glucose uptake rate (mmol/gDW/h)",font2)
    plt.ylabel("Growth rate ($\mathregular{h^-1}$)",font2)
    plt.savefig("./Analysis Result/max_product_by_four_model.png")
    plt.show()

def Draw_MDF_By_Growth_rate(MDF_list):
    plt.figure(figsize=(15, 10), dpi=300)
    MDF_list=MDF_list.sort_index(ascending=True) 
    plt.plot(MDF_list.index, MDF_list[MDF_list.columns[0]], color="cyan", linewidth=3.0, linestyle="-", label=MDF_list.columns[0])
    plt.plot(MDF_list.index, MDF_list[MDF_list.columns[1]], color="darkorange", linewidth=3.0, linestyle="-", label=MDF_list.columns[1])
    font1 = {'family' : 'Times New Roman',
    'weight' : 'normal',
    'size'   : 23,
    }

    font2 = {'family' : 'Times New Roman',
    'weight' : 'normal',
    'size'   : 30,
    }

    plt.ylabel("MDF of pathways (kJ/mol)",font2)
    plt.xlabel("Growth rate ($\mathregular{h^-1}$)",font2)

    ax = plt.gca()
    ax.spines['right'].set_color('none')
    ax.spines['top'].set_color('none')
    ax.spines['left'].set_position(('data', 0.3))
    ax.spines['bottom'].set_position(('data', 0))
    plt.legend(loc="lower left",prop=font1)
    plt.xlim(0.3, 0.9)
    plt.ylim(-26, 3)

    plt.tick_params(labelsize=23)

    plt.xticks([0.3, 0.4, 0.5,0.6, 0.7,0.8, 0.9])
    plt.yticks([-26, -22, -18,-14, -10,-6, -2,2])

    #plt.scatter([0.633], [2.6670879363966336], s=80, color="red")
    #plt.scatter([0.6756], [-0.48186643213771774], s=80, color="red")
    #plt.scatter([0.7068], [-9.486379882991386], s=80, color="red")
    #plt.scatter([0.852], [2.6670879363966336], s=80, color="red")
    #plt.scatter([0.855], [1.4290141211096987], s=80, color="red")
    #plt.scatter([0.867], [0.06949515162540898], s=80, color="red")
    #plt.scatter([0.872], [-0.8364187795859692], s=80, color="red")
    #plt.scatter([0.876], [-9.486379882991372], s=80, color="red")

    plt.savefig("./Analysis Result/max_MDF_by_four_model.png")
    plt.show()
