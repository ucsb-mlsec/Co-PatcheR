import libcst as cst
from libcst.metadata import PositionProvider
from libcst.metadata import MetadataWrapper

def get_function_interval(file_content):
    module = cst.parse_module(file_content)
    wrapper = MetadataWrapper(module)

    class FunctionCollector(cst.CSTVisitor):
        METADATA_DEPENDENCIES = (PositionProvider,)

        def __init__(self):
            self.functions = {}

        def visit_FunctionDef(self, node):
            name = node.name.value
            pos = self.get_metadata(PositionProvider, node)
            start_line = pos.start.line
            end_line = pos.end.line
            self.functions[name] = (start_line, end_line) 
    collector = FunctionCollector()
    wrapper.visit(collector)
    return collector.functions
