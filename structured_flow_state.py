from crewai.flow.flow import Flow ,start,listen
from pydantic import BaseModel ,Field
from typing import Any

# define the pydantic model
class PersonState(BaseModel):
    name:str = Field(description='name of the person',default='')
    age : int = Field(description='Age of the person',default=0)
    friend_list : list[str] = Field(description='Friend list fo the person',default_factory=list)
    contact_info : dict[str,str|int] = Field(description='Conatct information',default={})
# now to pass the pydantic model we have to just give it as name to the flow state as below
# define the class
class StructuredFlowState(Flow[PersonState]):
    # defien first node
    @start()

    def get_inputs(self):
        # retrun the state
        print(f'state is {self.state}')
        return self.state
    

    @listen(get_inputs)
    def update_state(self):
        self.state.friend_list.append('khan')
        self.state.contact_info['email'] = 'abc@123'
        return 'State updated'

if __name__ == "__main__":

    flow = StructuredFlowState()

    result = flow.kickoff({
        'name':'Aamir',
        'age':33,
        'friend_list':['yaqoob','Hadi'],
        'contact_info': {'email':'khan@122334'}
    })

    print(f"output is {result}")

    # print the state

    print(f"Flow State : {flow.state}")