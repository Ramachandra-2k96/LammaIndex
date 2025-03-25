from llama_index.llms.ollama import Ollama
from llama_index.core.tools import FunctionTool
from llama_index.core.agent import ReActAgent
from llama_index.core.settings import Settings
from typing import List, Dict, Any
import datetime
from dateutil.parser import parse
from llama_index.core import PromptTemplate

class InterviewScheduler:
    def __init__(self, model_name="llama3.2:latest"):
        # Slot duration in hours
        self.slot_duration = 3
        
        # Hardcoded interviewer and candidate
        self.interviewer = "HR"
        self.candidate = "John Smith"
        
        # Initialize calendars with pre-populated events
        self.interviewer_calendar = self._initialize_interviewer_calendar()
        self.candidate_calendar = self._initialize_candidate_calendar()
        
        # Initialize booked interviews list
        self.booked_interviews = []
        
        # Set up LlamaIndex with Ollama
        self.llm = Ollama(model=model_name, request_timeout=420.0)
        
        # Initialize settings with the LLM
        Settings.llm = self.llm
        Settings.chunk_size = 2024
        
        # System prompt for the agent
        react_system_header_str = """\
        You are an intelligent interview scheduling assistant designed to help schedule, reschedule, and manage interviews.

        ## Available Participants
        - Interviewer: {interviewer} [You are his assisatant]
        - Candidate: {candidate} [This is the User you are talking with]
        - Interview slots are {slot_duration} hours long, from {start_time} to {end_time}
        - Available slot start times are {available_slots}

        ## Tools
        You have access to the following specialized tools to manage interview scheduling. You are responsible for using these tools in the appropriate sequence to complete scheduling tasks.

        {tool_desc}

        ## Output Format
        To process a user request, please use the following format:
        Thought: I need to analyze the user's request and determine which tool to use.
        Action: tool name (one of {tool_names})
        Action Input: the input to the tool in valid JSON format
        Always start with a Thought.

        If the user's input is vague or unclear (e.g., "hello"), respond with:
        Thought: The user's request is not clear enough to take action.
        Answer: Hello! I can help you schedule interviews between HR and John Smith. You can:
        - Schedule an interview
        - Reschedule an existing interview
        - Cancel an interview
        - Check available slots
        - List scheduled interviews
        What would you like to do?

        Ensure you use proper JSON format for Action Input with double quotes around keys and string values.

        After using a tool, you will receive a response in this format:
        Observation: tool response
        Continue this process until you have sufficient information to fully address the user's request. When you have everything needed, respond in this format:
        Thought: I can answer without using any more tools.
        Answer: [your detailed response here]

        ## Key Guidelines
        - Always check availability before attempting to schedule an interview
        - Confirm all details before executing scheduling actions
        - Use a friendly, professional tone in all responses
        - Format dates as YYYY-MM-DD and times in 24-hour format (HH:MM)
        - Only schedule within available slots ({available_slots})
        - When listing available slots or interviews, format them clearly for easy reading
        - Maintain context across the conversation by recalling previously mentioned information
        - Ask clarifying questions when information is incomplete
        - Today's date is {today_date}

        ## Response Structure
        Your final answer should:
        - Confirm the action taken (scheduling, rescheduling, cancellation, etc.)
        - Include all relevant details (date, time, participants, interview ID if applicable)
        - Suggest next steps when appropriate
        - Use a clear, organized format with bullet points for important information

        ## Current Conversation
        Below is the current conversation consisting of interleaving human and assistant messages.
"""
        
        # Define tools for the agent
        self.tools = [
            FunctionTool.from_defaults(
                fn=self.get_available_slots,
                name="get_available_slots",
                description="Get next 10 available interview slots. Use when you need to check availability."
            ),
            FunctionTool.from_defaults(
                fn=self.schedule_interview,
                name="schedule_interview",
                description="Schedule an interview at a specific slot (YYYY-MM-DD HH:MM)."
            ),
            FunctionTool.from_defaults(
                fn=self.reschedule_interview,
                name="reschedule_interview",
                description="Reschedule an existing interview. Requires interview ID and new slot (YYYY-MM-DD HH:MM)."
            ),
            FunctionTool.from_defaults(
                fn=self.cancel_interview,
                name="cancel_interview",
                description="Cancel a scheduled interview. Requires interview ID."
            ),
            FunctionTool.from_defaults(
                fn=self.list_scheduled_interviews,
                name="list_scheduled_interviews",
                description="List all scheduled interviews."
            )
        ]
        
        # Initialize agent
        self.agent = ReActAgent.from_tools(
            self.tools,
            llm=self.llm,
            verbose=True,
            max_iterations=10,
            is_agent_observation_separated=True
        )
        # tool_descriptions = ""
        # tool_names = []

        # for tool in self.tools:
        #     tool_descriptions += f"- {tool.metadata.name}: {tool.metadata.description}\n"
        #     tool_names.append(tool.metadata.name)

        # tool_names_str = ", ".join([f"'{name}'" for name in tool_names])
        # # Format the system prompt with dynamic values
        # available_slot_times = ", ".join([f"{hour:02d}:00" for hour in range(9, 17, self.slot_duration)])
        # formatted_system_prompt = react_system_header_str.format(
        #     interviewer=self.interviewer,
        #     candidate=self.candidate,
        #     slot_duration=self.slot_duration,
        #     start_time="9:00",
        #     end_time="17:00",
        #     available_slots=available_slot_times,
        #     tool_desc=tool_descriptions,
        #     tool_names=tool_names_str,  # Use the string version here
        #     today_date=datetime.datetime.today().strftime("%Y-%m-%d")
        # )

        # # Use the formatted prompt
        # react_system_prompt = PromptTemplate(formatted_system_prompt)
        # self.agent.update_prompts({"agent_worker:system_prompt": react_system_prompt})
        # Store conversation context
        self.conversation_context = {
            "pending_action": None,
            "last_listed_interviews": [],
            "last_listed_slots": []
        }
    
    def _initialize_interviewer_calendar(self) -> Dict[str, Dict[str, bool]]:
        """Initialize HR's calendar with some pre-populated events"""
        calendar_data = {}
        today = datetime.datetime.now()
        
        for i in range(30):
            current_date = today + datetime.timedelta(days=i)
            date_str = current_date.strftime("%Y-%m-%d")
            
            # Initialize slots for each day (9AM to 5PM with 3-hour slots)
            slots = {}
            for hour in range(9, 17, self.slot_duration):
                slot_key = f"{hour:02d}:00"
                # Default to available
                slots[slot_key] = True
            
            calendar_data[date_str] = slots
        
        # Add hardcoded events to interviewer calendar
        # Representing meetings, other interviews, etc.
        
        # Tomorrow
        tomorrow = (today + datetime.timedelta(days=1)).strftime("%Y-%m-%d")
        calendar_data[tomorrow]["09:00"] = False  # Morning meeting
        calendar_data[tomorrow]["15:00"] = False  # Performance review
        
        # Day after tomorrow
        day_after = (today + datetime.timedelta(days=2)).strftime("%Y-%m-%d")
        calendar_data[day_after]["12:00"] = False  # Lunch meeting
        
        # Next week
        next_week = (today + datetime.timedelta(days=7)).strftime("%Y-%m-%d")
        calendar_data[next_week]["09:00"] = False  # Team meeting
        calendar_data[next_week]["12:00"] = False  # Department meeting
        calendar_data[next_week]["15:00"] = False  # Training session
        
        return calendar_data
    
    def _initialize_candidate_calendar(self) -> Dict[str, Dict[str, bool]]:
        """Initialize candidate's calendar with some preferences/constraints"""
        calendar_data = {}
        today = datetime.datetime.now()
        
        for i in range(30):
            current_date = today + datetime.timedelta(days=i)
            date_str = current_date.strftime("%Y-%m-%d")
            
            # Initialize slots for each day
            slots = {}
            for hour in range(9, 17, self.slot_duration):
                slot_key = f"{hour:02d}:00"
                # Default to available
                slots[slot_key] = True
            
            calendar_data[date_str] = slots
        
        # Add candidate's unavailable times
        
        # Tomorrow
        tomorrow = (today + datetime.timedelta(days=1)).strftime("%Y-%m-%d")
        calendar_data[tomorrow]["12:00"] = False  # Appointment
        
        # Next week
        next_week = (today + datetime.timedelta(days=5)).strftime("%Y-%m-%d")
        calendar_data[next_week]["09:00"] = False  # Personal commitment
        calendar_data[next_week]["12:00"] = False  # Personal commitment
        
        return calendar_data
    
    def get_available_slots(self) -> Dict[str, List[str]]:
        """
        Retrieve the next 10 available interview slots for both the interviewer and candidate.

        This function iterates over the interviewer’s and candidate’s calendars, checking for common available
        time slots. It collects available slots in the format "YYYY-MM-DD HH:MM" and stops once 10 slots have been found.
        The list of found slots is also stored in the conversation context for future reference.

        Returns:
            dict: A dictionary with a key "available_slots" that maps to a list of available slot strings.
        """
        available_slots = []
        today = datetime.datetime.now()
        
        # Loop through dates in the calendar
        for date_str in sorted(self.interviewer_calendar.keys()):
            date_obj = datetime.datetime.strptime(date_str, "%Y-%m-%d")
            
            # Skip past dates
            if date_obj.date() < today.date():
                continue
            
            interviewer_slots = self.interviewer_calendar[date_str]
            candidate_slots = self.candidate_calendar[date_str]
            
            # Check each time slot
            for time_str, interviewer_available in interviewer_slots.items():
                candidate_available = candidate_slots.get(time_str, False)
                
                # Only include slots where both parties are available
                if interviewer_available and candidate_available:
                    slot_full_str = f"{date_str} {time_str}"
                    available_slots.append(slot_full_str)
                    
                    # Return once we have 10 slots
                    if len(available_slots) >= 10:
                        self.conversation_context["last_listed_slots"] = available_slots
                        return {"available_slots": available_slots}
        
        self.conversation_context["last_listed_slots"] = available_slots
        return {"available_slots": available_slots}


    def schedule_interview(self, slot: str) -> Dict[str, Any]:
        """
        Schedule an interview at a specific slot.

        This function attempts to book an interview slot by taking a string input in the "YYYY-MM-DD HH:MM" format.
        It parses and rounds the provided time to the nearest valid slot (based on the slot duration) and then checks
        both the interviewer’s and candidate’s calendars to ensure that both parties are available at that time.
        Upon successful scheduling, the function marks the slot as unavailable and records the interview details.

        Args:
            slot (str): The desired interview slot in the "YYYY-MM-DD HH:MM" format.

        Returns:
            dict: A dictionary containing:
                - "success": A boolean indicating whether the scheduling was successful.
                - "message": A success message including the interview ID, or an error message.
                - "details": (Optional) A dictionary with the interview details if scheduling succeeded.
                - "error": (Optional) An error message if scheduling failed.
        """
        # Parse the slot string
        try:
            slot_datetime = parse(slot)
            date_str = slot_datetime.strftime("%Y-%m-%d")
            time_str = slot_datetime.strftime("%H:%M")
            
            # Round time to the nearest available slot time
            hour = int(time_str.split(":")[0])
            valid_hours = list(range(9, 17, self.slot_duration))
            closest_hour = min(valid_hours, key=lambda x: abs(x - hour))
            time_str = f"{closest_hour:02d}:00"
            
        except ValueError:
            return {"error": "Invalid date format. Please use YYYY-MM-DD HH:MM format."}
        
        # Check if the date is in the calendar
        if date_str not in self.interviewer_calendar or date_str not in self.candidate_calendar:
            return {"error": f"Date {date_str} is not available in the calendar"}
        
        # Check if the time slot is valid
        if time_str not in self.interviewer_calendar[date_str] or time_str not in self.candidate_calendar[date_str]:
            return {"error": f"Time slot {time_str} is not available. Available slots are at 09:00, 12:00, and 15:00"}
        
        # Check if both parties are available
        if not self.interviewer_calendar[date_str][time_str]:
            return {"error": f"Interviewer {self.interviewer} is not available at {date_str} {time_str}"}
        
        if not self.candidate_calendar[date_str][time_str]:
            return {"error": f"Candidate {self.candidate} is not available at {date_str} {time_str}"}
        
        # Book the slot
        self.interviewer_calendar[date_str][time_str] = False
        self.candidate_calendar[date_str][time_str] = False
        
        # Record the interview
        interview_id = len(self.booked_interviews) + 1
        interview_record = {
            "id": interview_id,
            "candidate": self.candidate,
            "interviewer": self.interviewer,
            "date": date_str,
            "time": time_str,
            "end_time": f"{(int(time_str.split(':')[0]) + self.slot_duration):02d}:00"
        }
        self.booked_interviews.append(interview_record)
        
        return {
            "success": True,
            "message": f"Interview scheduled with ID #{interview_id}",
            "details": interview_record
        }


    def reschedule_interview(self, interview_id: int, new_slot: str) -> Dict[str, Any]:
        """
        Reschedule an existing interview to a new slot.

        This function looks up an existing interview using its interview ID and attempts to reschedule it to a new slot
        provided in the "YYYY-MM-DD HH:MM" format. It frees up the previously booked slot and then attempts to book the new slot.
        If booking the new slot fails, the original booking is reverted.

        Args:
            interview_id (int): The unique identifier of the interview to be rescheduled.
            new_slot (str): The new desired interview slot in the "YYYY-MM-DD HH:MM" format.

        Returns:
            dict: A dictionary containing:
                - "success": A boolean indicating whether the rescheduling was successful.
                - "message": A message detailing the outcome of the rescheduling.
                - "old_details": A dictionary with the original interview details.
                - "new_details": (Optional) A dictionary with the new interview details if rescheduling succeeded.
                - "error": (Optional) An error message if rescheduling failed.
        """
        # Convert interview_id to int if necessary
        try:
            interview_id = int(interview_id)
        except ValueError:
            return {"error": "Interview ID must be a number"}
                
        # Find the interview
        interview = None
        interview_index = None
        for i, interview_record in enumerate(self.booked_interviews):
            if interview_record["id"] == interview_id:
                interview = interview_record
                interview_index = i
                break
        
        if not interview:
            return {"error": f"Interview with ID {interview_id} not found"}
        
        # Free up the current slot
        date_str = interview["date"]
        time_str = interview["time"]
        
        self.interviewer_calendar[date_str][time_str] = True
        self.candidate_calendar[date_str][time_str] = True
        
        # Attempt to schedule the new slot
        result = self.schedule_interview(new_slot)
        
        if "error" in result:
            # Revert the cancellation if scheduling fails
            self.interviewer_calendar[date_str][time_str] = False
            self.candidate_calendar[date_str][time_str] = False
            return result
        
        # Remove the old interview record
        self.booked_interviews.pop(interview_index)
        
        return {
            "success": True,
            "message": f"Interview with ID #{interview_id} rescheduled",
            "old_details": interview,
            "new_details": result.get("details")
        }


    def cancel_interview(self, interview_id: int) -> Dict[str, Any]:
        """
        Cancel a scheduled interview.

        This function cancels an existing interview identified by its interview ID. It marks the corresponding slot
        in both the interviewer’s and candidate’s calendars as available again and removes the interview record from
        the list of booked interviews.

        Args:
            interview_id (int): The unique identifier of the interview to cancel.

        Returns:
            dict: A dictionary containing:
                - "success": A boolean indicating whether the cancellation was successful.
                - "message": A success message including the interview ID if cancellation was successful.
                - "details": A dictionary containing the details of the cancelled interview.
                - "error": (Optional) An error message if cancellation failed.
        """
        # Convert interview_id to int if necessary
        try:
            interview_id = int(interview_id)
        except ValueError:
            return {"error": "Interview ID must be a number"}
                
        # Find the interview
        interview = None
        interview_index = None
        for i, interview_record in enumerate(self.booked_interviews):
            if interview_record["id"] == interview_id:
                interview = interview_record
                interview_index = i
                break
        
        if not interview:
            return {"error": f"Interview with ID {interview_id} not found"}
        
        # Free up the slot
        date_str = interview["date"]
        time_str = interview["time"]
        
        self.interviewer_calendar[date_str][time_str] = True
        self.candidate_calendar[date_str][time_str] = True
        
        # Remove the interview record
        self.booked_interviews.pop(interview_index)
        
        return {
            "success": True,
            "message": f"Interview with ID #{interview_id} cancelled",
            "details": interview
        }


    def list_scheduled_interviews(self) -> Dict[str, List[Dict]]:
        """
        List all scheduled interviews.

        This function compiles and returns a list of all currently booked interviews along with their details.
        It also updates the conversation context to store the list of interviews for subsequent interactions.

        Returns:
            dict: A dictionary with a key "scheduled_interviews" that maps to a list of dictionaries,
                each representing a scheduled interview and its details.
        """
        self.conversation_context["last_listed_interviews"] = self.booked_interviews
        return {"scheduled_interviews": self.booked_interviews}

    
    def process_query(self, query: str) -> str:
        """Process a natural language query using the agent"""
        # Add context from previous interactions if available
        context_info = ""
        if self.conversation_context["last_listed_slots"]:
            context_info += f"Recently viewed slots: {', '.join(self.conversation_context['last_listed_slots'][:3])}. "
        if self.conversation_context["last_listed_interviews"]:
            interview_ids = [str(interview['id']) for interview in self.conversation_context['last_listed_interviews']]
            context_info += f"Recently viewed interview IDs: {', '.join(interview_ids[:3])}. "
        
        # Enhance query with context if meaningful
        if context_info:
            query = f"Context: {context_info}\n\nUser query: {query}"
        
        response = self.agent.chat(query)
        
        # Track potential pending actions
        if "schedule" in query.lower() and "confirm" not in query.lower():
            self.conversation_context["pending_action"] = "schedule"
        elif "reschedule" in query.lower() and "confirm" not in query.lower():
            self.conversation_context["pending_action"] = "reschedule"
        
        return response.response

def run_cli_interface():
    """Run a simple CLI interface for the interview scheduler"""
    try:
        print("Interview Scheduling System")
        print("===========================")
        print("Type 'exit' to quit the application\n")
        print("You can ask to:")
        print("- Schedule interviews between HR and John Smith")
        print("- Reschedule interviews")
        print("- Cancel interviews")
        print("- Check available slots")
        print("- List scheduled interviews\n")
        
        scheduler = InterviewScheduler()
        
        while True:
            query = input("\nWhat would you like to do? > ")
            if query.lower() == 'exit':
                break
            
            try:
                response = scheduler.process_query(query)
                print(f"\nResponse: {response}")
            except Exception as e:
                print(f"\nError processing query: {str(e)}")
                
    except Exception as e:
        print(f"Error initializing scheduler: {str(e)}")
            
# Example usage with command line interface
if __name__ == "__main__":
    run_cli_interface()