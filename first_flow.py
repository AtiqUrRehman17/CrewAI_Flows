# imports
from crewai.flow.flow import Flow, start ,listen

# define the flow class

class MyFirstFlow(Flow): # inherit form the flow class

    # define the start node which will start the flow
    @start
    def start_flow(self) -> str:
        return"Flow started Successfully"
    
    # make the second node and connect it to the first node
    @listen(start_flow) # just pass the first node name 
    def processing_node(self,output) -> str:
        print(f"output of start node ;{output}")
        return"Data Process Complete"
    
    @listen(processing_node)
    def flow_complete(self,output_2) -> str:
        print(f"output of the node 2 : {output_2}")
        return "flow completed"
    
if __name__ == "__main__":

    # create the flow
    flow = MyFirstFlow()

    # execute the flow 
    result = flow.kickoff()

    # plot the flow
    flow.plot('first_flow.html')

    # print the result
    print(result)

