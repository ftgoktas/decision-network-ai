import pyAgrum as gum
import pyAgrum.lib.image as gumimage

# Create a decision network
model = gum.InfluenceDiagram()

# Add a decision node for Cheat 1
Cheat_1 = gum.LabelizedVariable('Cheat 1', 'Cheat 1', 2)
Cheat_1.changeLabel(0, 'False')
Cheat_1.changeLabel(1, 'True')
model.addDecisionNode(Cheat_1)

# Add a decision node for Cheat 2
Cheat_2 = gum.LabelizedVariable('Cheat 2', 'Cheat 2', 2)
Cheat_2.changeLabel(0, 'False')
Cheat_2.changeLabel(1, 'True')
model.addDecisionNode(Cheat_2)

# Add a chance node for result of Trouble 1
result = gum.LabelizedVariable('Trouble 1', 'Trouble 1', 2)
result.changeLabel(0, 'False')
result.changeLabel(1, 'True')
model.addChanceNode(result)

# Add a chance node for result of Trouble 2
result = gum.LabelizedVariable('Trouble 2', 'Trouble 2', 2)
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

# Add utilities
model.utility(model.idFromName('Utility'))[{'Trouble 2': 'True', 'Cheat 2': 'True'}] = -30
model.utility(model.idFromName('Utility'))[{'Trouble 2': 'True', 'Cheat 2': 'False'}] = 70
model.utility(model.idFromName('Utility'))[{'Trouble 2': 'False', 'Cheat 2': 'True'}] = -70
model.utility(model.idFromName('Utility'))[{'Trouble 2': 'False', 'Cheat 2': 'False'}] = 100

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
