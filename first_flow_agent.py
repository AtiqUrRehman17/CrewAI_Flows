from crewai.flow.flow import Flow ,start, listen
from crewai import Agent
from dotenv import load_dotenv
from pydantic import BaseModel,Field

# define the structured output
class CityName(BaseModel):
    city_name: str = Field(description='Name of the City')
    country_name : str = Field(description='Name of the Country')


# load the apis to env

load_dotenv()

# defin the flow class 
class MyFirstAgent(Flow):

    # define the first nodel (start)

    @start()
    def generate_city_name(self) -> dict[str,str]:
        # now to define agent
        city_agent = Agent(
            role = "Expert in City name and Country name in the World",
            goal = 'You have To Generate a random city name and courty name of the City',
            backstory = 'You have to been created to generate Random names of the City and Country of that city.You have been working this for 3 years.'
        )
        # give a prompt
        prompt = 'Generate a City name and also the Country name of that city'
        # get the oupput
        agent_output = city_agent.kickoff(messages=prompt,
                                          response_format=CityName)
        
        # print the Dict
        print(f'City name output is : {agent_output.pydantic.model_dump()}')
        # return the output as dict
        return agent_output.pydantic.model_dump()
    
    # define the second node
    @listen(generate_city_name)

    def generate_fact(self,output_node) ->str:
        # now to define the second agent
        city_name = output_node["city_name"]
        country_name = output_node["country_name"]
        fact_agent = Agent(
            role = 'Expert in generate a fun fact about the city and its corresponding counrty',
            goal = f'You have been tasked to generate a fun fact about the city : {city_name} and its corresponding country {country_name}.the fun fact should be funny.',
            backstory = f'You have been generating fun facts about a city name :{city_name} and its corresponding country name :{country_name}.you have been doing this for 2 years and also the fun facts are very funny and the users like it very at all.'
        )
        # define the input propt
        prompt = f"generate a fun fact about the city {city_name} and the fact should be fun to read also include the name of the country {country_name}in the output"

        # exctute the agent
        fact = fact_agent.kickoff(prompt)
        return fact.raw
    
if __name__ == "__main__":
    # create the flow
    flow = MyFirstAgent()

    # execute the flow
    result = flow.kickoff()
    # plot the flow

    flow.plot('agent_flow.html')

    print(f"Flow Output :\n {result}")