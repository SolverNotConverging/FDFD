if(NOT DEFINED INSPECTOR OR NOT DEFINED FILE OR NOT DEFINED SCENARIO)
    message(FATAL_ERROR "INSPECTOR, FILE, and SCENARIO are required")
endif()

if(SCENARIO STREQUAL "single")
    set(arguments "${FILE}" 0 1 --coefficients)
    set(required
        "loaded_field_samples=2"
        "loaded_coefficients=4")
elseif(SCENARIO STREQUAL "sweep")
    set(arguments "${FILE}" 1 1 --coefficients)
    set(required
        "kind=sweep"
        "cases=2"
        "total_modes=4"
        "selected_case=1"
        "selected_mode=1"
        "frequency_hz=11000000000"
        "loaded_coefficients=4")
elseif(SCENARIO STREQUAL "3d")
    set(arguments "${FILE}" 0 0)
    set(required
        "dimension=3"
        "topology=tetra4"
        "loaded_field_samples=1")
else()
    message(FATAL_ERROR "Unknown inspector-check scenario: ${SCENARIO}")
endif()

execute_process(
    COMMAND "${INSPECTOR}" ${arguments}
    RESULT_VARIABLE result
    OUTPUT_VARIABLE output
    ERROR_VARIABLE error
)
if(NOT result EQUAL 0)
    message(FATAL_ERROR "Inspector exited ${result}: ${error}")
endif()
string(REPLACE "\r\n" "\n" normalized_output "${output}")
string(REPLACE "\r" "\n" normalized_output "${normalized_output}")
string(REGEX REPLACE "\n$" "" normalized_output "${normalized_output}")
string(REPLACE "\n" ";" output_lines "${normalized_output}")
foreach(required_line IN LISTS required)
    list(FIND output_lines "${required_line}" line_index)
    if(line_index EQUAL -1)
        message(FATAL_ERROR
            "Inspector output is missing exact line ${required_line}. Full output:\n${output}")
    endif()
endforeach()
