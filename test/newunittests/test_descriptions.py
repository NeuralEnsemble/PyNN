from pyNN import common, errors, random, standardmodels, space, descriptions
from nose.tools import assert_equal
import numpy
from mock import Mock
import os.path
    
class MockTemplateEngine(descriptions.TemplateEngine):
    render = Mock(return_value="african swallow")
    get_template = Mock()
    
def test_get_default_template_engine():
    engine = descriptions.get_default_template_engine()
    assert issubclass(engine, descriptions.TemplateEngine)

def test_render_with_no_template():
    context = {'a':2, 'b':3}
    result = descriptions.render(Mock(), None, context)
    assert_equal(result, context)

def test_render_with_template():
    engine = MockTemplateEngine
    context = {'a':2, 'b':3}
    template = "abcdefg"
    result = descriptions.render(engine, template, context)
    engine.render.assert_called_with(template, context)
    assert_equal(result, "african swallow")

def test_StringTE_get_template():
    result = descriptions.StringTemplateEngine.get_template("$a $b c d")
    assert_equal(result.template, "$a $b c d")

def test_StringTE_get_template_from_file():
    filename = "population_default.txt"
    result = descriptions.StringTemplateEngine.get_template(filename)
    assert result.template != filename
   
def test_StringTE_render():
    context = {'a':2, 'b':3}
    result = descriptions.StringTemplateEngine.render("$a $b c d", context)
    assert_equal(result, "2 3 c d")
    
def test_Jinja2TE_get_template_from_file():
    filename = "population_default.txt"
    result = descriptions.Jinja2TemplateEngine.get_template(filename)
    assert_equal(os.path.basename(result.filename), filename)
    
def test_Jinja2TE_render():
    context = {'a':2, 'b':3}
    result = descriptions.Jinja2TemplateEngine.render("{{a}} {{b}} c d", context)
    assert_equal(result, "2 3 c d")
    
def test_CheetahTE_get_template_from_file():
    filename = "population_default.txt"
    result = descriptions.CheetahTemplateEngine.get_template(filename)
    # incomplete test
    
def test_CheetahTE_render():
    context = {'a':2, 'b':3}
    result = descriptions.CheetahTemplateEngine.render("$a $b c d", context)
    assert_equal(result, "2 3 c d")
    
