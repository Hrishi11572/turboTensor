from .parameter import Parameter 

class Module: 
    
    def __init__(self): 
        self._parameters = {} 
        self._modules = {}
    
    def register_parameter(self, name, parameter): 
        self._parameters[name] = parameter 
    
    def add_module(self, name, module):
        self._modules[name] = module

    def parameters(self): 
        
        params = list(self._parameters.values())
        
        for module in self._modules.values(): 
            params.extend(module.parameters())
        
        return params 

    def state_dict(self):

        state = {}

        for name, parameter in self._parameters.items():
            state[name] = parameter.data.copy()

        for name, module in self._modules.items():

            child_state = module.state_dict()

            for child_name, value in child_state.items():
                state[f"{name}.{child_name}"] = value

        return state
    
    def load_state_dict(self, state):

        for name, parameter in self._parameters.items():

            if name not in state:
                raise KeyError(
                    f"Missing parameter: {name}"
                )

            parameter.data[...] = state[name]

        for name, module in self._modules.items():

            prefix = name + "."

            child_state = {
                key[len(prefix):]: value
                for key, value in state.items()
                if key.startswith(prefix)
            }

            module.load_state_dict(child_state)
                
    def to(self, device):
        for parameter in self._parameters.values():
            parameter.to(device)

        for module in self._modules.values():
            module.to(device)

        return self
    
    

# class Module:

#     def parameters(self):
#         params = []

#         for value in self.__dict__.values():
#             if isinstance(value, Parameter):
#                 params.append(value)

#             elif isinstance(value, Module):
#                 params.extend(value.parameters())

#             elif isinstance(value, (list, tuple)): 
#                 for item in value: 
#                     if isinstance(item, Parameter):
#                         params.append(item)

#                     elif isinstance(item, Module): 
#                         params.extend(item.parameters())
                
#         return params