from keras.models import Model, Sequential

class Architecture:
    def __init__(self, input_shape: tuple, num_classes: int):
        super().__init__()
        self.input_shape = input_shape
        self.num_classes = num_classes

    def build(self) -> Model:
        pass

    def name(self) -> str:
        return self.__class__.__name__

    def __str__(self):
        return self.name()

    def __repr__(self):
        return self.name()

    def build_sequential(self, layers: list, name:str=None) -> Sequential:
        if name is None:
            name = self.name()
        return Sequential(layers, name)

    def build_model(self, inputs, outputs, name:str=None) -> Model:
        if name is None:
            name = self.name()
        return Model(inputs=inputs, outputs=outputs, name=name)
