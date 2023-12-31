export const enum FieldType {
    RANGE = "range",
    SEED = "seed",
    TEXTAREA = "textarea",
    CHECKBOX = "checkbox",
    SELECT = "select",
}
export const enum PipelineMode {
    IMAGE = "image",
    VIDEO = "video",
    TEXT = "text",
}


export interface Fields {
    [key: string]: FieldProps;
}

export interface FieldProps {
    default: number | string;
    max?: number;
    min?: number;
    title: string;
    field: FieldType;
    step?: number;
    disabled?: boolean;
    hide?: boolean;
    id: string;
    values?: string[];
}
export interface PipelineInfo {
    title: {
        default: string;
    }
    name: string;
    description: string;
    input_mode: {
        default: PipelineMode;
    }
}