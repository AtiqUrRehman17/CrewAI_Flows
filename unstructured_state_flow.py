from crewai.flow.flow import Flow, start,listen
from typing import Any
# define the class of the flow
class UnstructuredStateFlow(Flow):

    # define the start node
    @start()
    def get_input(self) -> dict[str,Any]:

        # to access the state --> use self.state state is inside the class
        # unstructured state is in the form of {} 
        # lets see hoe to edit the state
        self.state['name'] = 'Atiq'
        self.state['Age'] = 25
        self.state['Friend_list'] = ['Atiq','Hassan','Fahim']
        return self.state
    

    @listen(get_input)
    def update_state(self,last_output) -> str:
        print('Last_output',last_output)

        return "State Updated"


if __name__ == "__main__":

    flow = UnstructuredStateFlow()

    # execute the flow

    result = flow.kickoff()

    # print
    print(f"Output is :{result}")