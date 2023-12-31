<script lang="ts">
  import type { FieldProps } from '$lib/types';
  import { onMount } from 'svelte';
  export let value = false;
  export let params: FieldProps;
  onMount(() => {
    value = Boolean(params?.default) ?? 8.0;
  });
</script>

<div class="grid max-w-md grid-cols-4 items-center justify-items-start gap-3">
  <label class="text-sm font-medium" for={params.id}>{params?.title}</label>
  <input bind:checked={value} type="checkbox" id={params.id} class="cursor-pointer" />
</div>
