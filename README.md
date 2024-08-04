# Problem Statement
Wild animals can present a distinct threat to safety and security, whether of individuals, livestock, or assets.  As such, early detection is crucial.  A Convolutional Neural Network (CNN) model can provide helpful aid in this endeavor, detecting the presence of a wild animal, in addition to whether it presents a danger.  The differentiation of dangerous animals and harmless ones is left up to the user and may differ by his or her goals for the technology, as certain species may present a higher level of danger to human life, while others may pose a greater threat to livestock or assets.

# Deep Learning Model
Our chosen model is YOLO (You Only Look Once). We trained it using a custom
dataset, which includes images of a handful of different animals and their corresponding labels (Dangerous or
Not Dangerous).  For the purpose of this experiment, 7 animals were included: buffalo, elephant, rhino, zebra, lion, ostrich, turtle.  The "dangerous" label was conferred upon the buffalo, rhino, lion, and ostrich classes.

# Custom Dataset
[widlife_detector.zip](https://drive.google.com/uc?export=download&id=1FrPo0bICEH8Xwuyfl2TgBRycyQ6UYUtb)

# Modifying Model
### Modifying Animals
Additional animal image sets can be added externally, using sources such as https://universe.roboflow.com/search?q=lion (modify query as needed).  Add each new animal class (along with index and label) to ```config.yaml``` in ```wildlife_detector.zip```, ensuring that the label index (the first value in the label ```.txt``` file for each instance) is set accordingly.

### Modifying Labels
The labels are set in ```config.yaml``` within ```wildlife_detector.zip```, and can be reconfigured depending on the goals of the user.
