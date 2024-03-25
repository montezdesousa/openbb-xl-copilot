from typing import List, Optional
import json
from pydantic import BaseModel, Field, root_validator

with open("reference.json") as f:
    FUNCTIONS = json.load(f)


class Parameter(BaseModel):
    name: str
    type_: str
    value: str


class Function(BaseModel):
    name: str = Field(description="Name of the function.")
    parameters: List[Parameter] = Field(description="Parameters of the function.")

    # pylint: disable=no-self-argument
    @root_validator(pre=True)
    def validate_function(cls, values):
        """Validate the function."""
        values["name"] = values["name"].replace("OBB.", "")
        func = cls.to_endpoint(values["name"])
        if func not in FUNCTIONS:
            raise ValueError(f"Function '{func}' not found in reference.")

        return values

    @staticmethod
    def to_endpoint(name: str) -> str:
        """Check if the function is valid."""
        return "/" + "/".join(name.lower().split("."))


class FunctionMessage(BaseModel):
    reasoning: str = Field(description="Short reasoning, keep it to 1 sentence.")
    function: Optional[Function] = Field(description="Function to be executed.")

    @staticmethod
    def _get_schema(name: str, field: str) -> dict:
        """Get the schema of the function."""
        return FUNCTIONS[Function.to_endpoint(name)][field]

    def to_xl(self) -> str:
        """Translate the function to Excel."""
        snippet = ""
        if self.function:
            incoming = {p.name: p.value for p in self.function.parameters}
            reference = self._get_schema(self.function.name, "parameters")
            args = ""
            for p_name, p_schema in reference.items():
                if p_name in incoming:
                    if p_schema["type"] == "Text":
                        args += f'"{incoming[p_name]}",'
                    else:
                        args += f"{incoming[p_name]},"
                else:
                    args += ","
            args = args.strip(",")

            snippet += "```excel\n"
            snippet += f"=OBB.{self.function.name}({args})\n"
            snippet += "```\n"

        t = self.reasoning + "\n\n"
        t += snippet
        return t
