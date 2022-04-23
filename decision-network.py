import pyAgrum as gum
import pyAgrum.lib.image as gumimage

# Create a decision network
model = gum.InfluenceDiagram()

# Add a decision node for cheat1
Cheat_1 = gum.LabelizedVariable('Cheat1', 'Cheat1', 2)
Cheat_1.changeLabel(0, 'False')
Cheat_1.changeLabel(1, 'True')
model.addDecisionNode(Cheat_1)

# Add a decision node for cheat2
Cheat_2 = gum.LabelizedVariable('Cheat2', 'Cheat2', 2)
Cheat_2.changeLabel(0, 'False')
Cheat_2.changeLabel(1, 'True')
model.addDecisionNode(Cheat_2)

# Add a chance node for result of trouble2
result = gum.LabelizedVariable('Trouble1', 'Trouble1', 2)
result.changeLabel(0, 'False')
result.changeLabel(1, 'True')
model.addChanceNode(result)

# Add a chance node for result of trouble2
result = gum.LabelizedVariable('Trouble2', 'Trouble2', 2)
result.changeLabel(0, 'False')
result.changeLabel(1, 'True')
model.addChanceNode(result)

# Add a chance node for result of watched
result = gum.LabelizedVariable('Watched', 'Watched', 2)
result.changeLabel(0, 'False')
result.changeLabel(1, 'True')
model.addChanceNode(result)

# Add an utility node
utility = gum.LabelizedVariable('Utility', 'Utility', 1)
model.addUtilityNode(utility)

# Add connections between nodes
model.addArc(model.idFromName('Cheat1'), model.idFromName('Trouble1'))
model.addArc(model.idFromName('Cheat1'), model.idFromName('Cheat2'))
model.addArc(model.idFromName('Trouble1'), model.idFromName('Cheat2'))
model.addArc(model.idFromName('Trouble1'), model.idFromName('Trouble2'))
model.addArc(model.idFromName('Cheat2'), model.idFromName('Trouble2'))
model.addArc(model.idFromName('Cheat2'), model.idFromName('Utility'))
model.addArc(model.idFromName('Trouble2'), model.idFromName('Utility'))
model.addArc(model.idFromName('Watched'), model.idFromName('Trouble1'))
model.addArc(model.idFromName('Watched'), model.idFromName('Trouble2'))

# Add utilities
model.utility(model.idFromName('Utility'))[{'Trouble2': 'True', 'Cheat2': 'True'}] = -30
model.utility(model.idFromName('Utility'))[{'Trouble2': 'True', 'Cheat2': 'False'}] = 70
model.utility(model.idFromName('Utility'))[{'Trouble2': 'False', 'Cheat2': 'True'}] = -70
model.utility(model.idFromName('Utility'))[{'Trouble2': 'False', 'Cheat2': 'False'}] = 100

# CPT for watched
model.cpt(model.idFromName('Watched'))[0] = 0.3  # F
model.cpt(model.idFromName('Watched'))[1] = 0.7  # T

# CPT for trouble1
model.cpt(model.idFromName('Trouble1'))[{'Watched': 'T', 'Cheat1': 'Yes'}] = \
    [.2,  # F
     .8]  # T
model.cpt(model.idFromName('Trouble1'))[{'Watched': 'T', 'Cheat1': 'No'}] = \
    [1,  # F
     0]  # T
model.cpt(model.idFromName('Trouble1'))[{'Watched': 'F', 'Cheat1': 'Yes'}] = \
    [1,  # F
     0]  # T
model.cpt(model.idFromName('Trouble1'))[{'Watched': 'F', 'Cheat1': 'No'}] = \
    [1,  # F
     0]  # T

# CPT for trouble2
model.cpt(model.idFromName('Trouble2'))[{'Cheat2': 'Yes', 'Trouble1': 'T', 'Watched': 'T'}] = \
    [0,  # F
     1]  # T

model.cpt(model.idFromName('Trouble2')) \
    [{'Cheat2': 'Yes', 'Trouble1': 'T', 'Watched': 'F'}] = \
    [.7,  # F
     .3]  # T

model.cpt(model.idFromName('Trouble2')) \
    [{'Cheat2': 'Yes', 'Trouble1': 'F', 'Watched': 'T'}] = \
    [.2,  # F
     .8]  # T

model.cpt(model.idFromName('Trouble2')) \
    [{'Cheat2': 'Yes', 'Trouble1': 'F', 'Watched': 'F'}] = \
    [1,  # F
     0]  # T

model.cpt(model.idFromName('Trouble2')) \
    [{'Cheat2': 'No', 'Trouble1': 'T', 'Watched': 'T'}] = \
    [.7,  # F
     .3]  # T

model.cpt(model.idFromName('Trouble2')) \
    [{'Cheat2': 'No', 'Trouble1': 'T', 'Watched': 'F'}] = \
    [.7,  # F
     .3]  # T

model.cpt(model.idFromName('Trouble2')) \
    [{'Cheat2': 'No', 'Trouble1': 'F', 'Watched': 'T'}] = \
    [1,  # F
     0]  # T

model.cpt(model.idFromName('Trouble2')) \
    [{'Cheat2': 'No', 'Trouble1': 'F', 'Watched': 'F'}] = \
    [1,  # F
     0]  # T

# Create an inference model
ie = gum.ShaferShenoyLIMIDInference(model)

# export to pdf
gumimage.export(model, "DecisionNetwork.pdf")

# Make an inference with default evidence
print('--- Inference with default evidence ---')
ie.makeInference()
print('Best decision for Cheat 1: {0}'.format(ie.optimalDecision(model.idFromName('Cheat 1'))))
print('Utility Cheat 1: {0}'.format(ie.posteriorUtility(model.idFromName('Cheat 1'))))
print('Maximum Expected Utility (MEU) : {0}'.format(ie.MEU()))
