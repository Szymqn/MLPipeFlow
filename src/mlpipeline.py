class MLPipeFlow:
    def __init__(self, components):
        self.components = components

    def run(self):
        data = None
        for component in self.components:
            data = component.execute(data)
        return data
