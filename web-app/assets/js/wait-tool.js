/* This module requires j-query. */

const WAIT_TOOL_SELECTOR = ".wait-tool"
const HIDE_CLASS = 'd-none';
// And then create our markup:
const WAIT_TOOL_MARKUP = `
<div class="sk-circle wait-tool d-none">
    <div class="sk-circle1 sk-child"></div>
    <div class="sk-circle2 sk-child"></div>
    <div class="sk-circle3 sk-child"></div>
    <div class="sk-circle4 sk-child"></div>
    <div class="sk-circle5 sk-child"></div>
    <div class="sk-circle6 sk-child"></div>
    <div class="sk-circle7 sk-child"></div>
    <div class="sk-circle8 sk-child"></div>
    <div class="sk-circle9 sk-child"></div>
    <div class="sk-circle10 sk-child"></div>
    <div class="sk-circle11 sk-child"></div>
    <div class="sk-circle12 sk-child"></div>
</div>
`;

function showWaitTools() {
    $(WAIT_TOOL_SELECTOR).each(function() {
        $(this).removeClass(HIDE_CLASS);
    });
}

function hideWaitTools() {
    $(WAIT_TOOL_SELECTOR).each(function() {
        $(this).addClass(HIDE_CLASS);
    });
}
 
function createWaitTool()
{
    return WAIT_TOOL_MARKUP;
}

function showWaitTool(selector)
{
    showWaitTools();

    waitTool = createWaitTool();
    content = $(selector).html();
    
    $(selector).html(waitTool);

    showWaitTools();
    
    window.scrollTo(0, 0);

    return content;
} 

function hideWaitTool(selector, content)
{
    $(selector).html(content);

    hideWaitTools();

    window.scrollTo(0, 0);
}