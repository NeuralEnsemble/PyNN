try:
    import unittest2 as unittest
except ImportError:
    import unittest

from pyNN import common, errors, random, standardmodels, space, descriptions
import numpy
try:
    from unittest.mock import Mock
except ImportError:
    from mock import Mock
import os.path

class MockTemplateEngine(descriptions.TemplateEngine):
    render = Mock(return_value="african swallow")
    get_template = Mock()


class DescriptionTest(unittest.TestCase):

    def test_get_default_template_engine(self):
        engine = descriptions.get_default_template_engine()
        assert issubclass(engine, descriptions.TemplateEngine)
    
    def test_render_with_no_template(self):
        context = {'a':2, 'b':3}
        result = descriptions.render(Mock(), None, context)
        self.assertEqual(result, context)
    
    def test_render_with_template(self):
        engine = MockTemplateEngine
        context = {'a':2, 'b':3}
        template = "abcdefg"
        result = descriptions.render(engine, template, context)
        engine.render.assert_called_with(template, context)
        self.assertEqual(result, "african swallow")
    
    def test_StringTE_get_template(self):
        result = descriptions.StringTemplateEngine.get_template("$a $b c d")
        self.assertEqual(result.template, "$a $b c d")
    
    def test_StringTE_get_template_from_file(self):
        filename = "population_default.txt"
        result = descriptions.StringTemplateEngine.get_template(filename)
        self.assertNotEqual(result.template, filename)
       
    def test_StringTE_render(self):
        context = {'a':2, 'b':3}
        result = descriptions.StringTemplateEngine.render("$a $b c d", context)
        self.assertEqual(result, "2 3 c d")
    
    @unittest.skipUnless('jinja2' in descriptions.TEMPLATE_ENGINES, "Requires Jinja2")
    def test_Jinja2TE_get_template_from_file(self):
        filename = "population_default.txt"
        result = descriptions.Jinja2TemplateEngine.get_template(filename)
        self.assertEqual(os.path.basename(result.filename), filename)
    
    @unittest.skipUnless('jinja2' in descriptions.TEMPLATE_ENGINES, "Requires Jinja2")
    def test_Jinja2TE_render(self):
        context = {'a':2, 'b':3}
        result = descriptions.Jinja2TemplateEngine.render("{{a}} {{b}} c d", context)
        self.assertEqual(result, "2 3 c d")
    
    @unittest.skipUnless('cheetah' in descriptions.TEMPLATE_ENGINES, "Requires Cheetah")
    def test_CheetahTE_get_template_from_file(self):
        filename = "population_default.txt"
        result = descriptions.CheetahTemplateEngine.get_template(filename)
        # incomplete test
    
    @unittest.skipUnless('cheetah' in descriptions.TEMPLATE_ENGINES, "Requires Cheetah")
    def test_CheetahTE_render(self):
        context = {'a':2, 'b':3}
        result = descriptions.CheetahTemplateEngine.render("$a $b c d", context)
        self.assertEqual(result, "2 3 c d")


if __name__ == "__main__":
    unittest.main()
