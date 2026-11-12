package common

// ToolChoiceOption is the options-map key under which a caller may set the
// tool choice for one request: ToolChoiceAuto (the default) or ToolChoiceNone.
// It applies only when the request carries tools.
const ToolChoiceOption = "tool_choice"

const (
	// ToolChoiceAuto lets the model decide whether to call a tool.
	ToolChoiceAuto = "auto"
	// ToolChoiceNone keeps the tools declared but forbids the model from
	// calling any of them, so it must answer in text.
	ToolChoiceNone = "none"
)

// ToolChoice returns the tool choice requested in options: ToolChoiceNone
// when options[ToolChoiceOption] is "none", otherwise ToolChoiceAuto.
func ToolChoice(options map[string]any) string {
	if v, ok := options[ToolChoiceOption].(string); ok && v == ToolChoiceNone {
		return ToolChoiceNone
	}
	return ToolChoiceAuto
}
