import os
import re

def format_timing(s):
    hours = int(s // 3600)
    minutes = int((s % 3600) // 60)
    seconds = int(s % 60)
    hundredths = int((s * 100) % 100)

    return f"{hours:02}:{minutes:02}:{seconds:02}.{hundredths:02}"

def clean_text(text, drop_spk=True):
    def replacer(match):
        content = match.group(1)
        if content == "spk" or content.startswith("*"):
            return ''  # Remove tags like <spk>, <*a>, etc.
        else:
            return content  # Keep punctuation like '.', '?', ',' etc.
    text = re.sub(r'<(.*?)>', replacer, text)
    if drop_spk:
        text = text.replace('<spk>', '')
    text = text.replace(" ' ", ' ')
    text = text.replace('  ', ' ')
    return text

def printv(s, verbose):
    if verbose:
        print(s)

def format_output(result, segments=None, add_timing=False, batch=False, verbose=False):
    if add_timing:
        assert segments is not None, "format_output got add_timing=True but segments=False so cannot infer the timing"

    out_results = {}
    for k, v in result.items():
        if k == 'ctc':
            tag = 'ENCODER'
        else: # subtitle/verbatim
            tag = k.upper()

        out_results[k] = []

        if isinstance(v, list):
            if len(v) > 1:
                printv('%s:' % tag)
                for i, el in enumerate(v):
                    if batch:
                        if add_timing:
                            out = '[%s - %s] %s' % (format_timing(segments[i]['start']), format_timing(segments[i]['end']), clean_text(el))
                            printv(out, verbose)
                            out_results[k].append(out)
                        else:
                            out = '%s' % clean_text(el)
                            printv(out, verbose)
                            out_results[k].append(out)
                    else:
                        if add_timing:
                            out = '[%s - %s] %s' % (format_timing(segments[i]['start']), format_timing(segments[i]['end']))
                            printv(out, verbose)
                            out_results[k].append(out)
                        out = '   HYP %i: %s' % (i+1, clean_text(el))
                        printv(out, verbose)
                        out_results[k].append(out)
            else:
                if add_timing:
                    out = '[%s - %s] %s' % (format_timing(segments[0]['start']), format_timing(segments[0]['end']), clean_text(v[0]))
                    printv(tag + ': ' + out, verbose)
                    out_results[k].append(out)
                else:
                    out = '%s' % (clean_text(v[0]))
                    printv(tag + ': ' + out, verbose)
                    out_results[k].append(out)
        else:
            if add_timing:
                out = '[%s - %s] %s' % (format_timing(segments[0]['start']), format_timing(segments[0]['end']), clean_text(v))
                printv(tag + ': ' + out, verbose)
                out_results[k].append(out)
            else:
                out = '%s: %s' % (tag, clean_text(v))
                printv(out, verbose)
                out_results[k].append(out)

    if verbose:
        print('')  # empty line

    return out_results

def merge_batch_outputs(results):
    out_results = {}
    for res in results:
        for k, v in res.items():
            if k not in out_results:
                out_results[k] = [v]
            else:
                out_results[k].append(v)
    return out_results

def write_output_to_file(results, outdir, tag):
    for k, v in results.items():
        with open(os.path.join(outdir, tag+'_%s.txt' % k), 'w', encoding="utf-8") as f:
            if isinstance(v, list):
                for el in v:
                    if isinstance(el, list):
                        for s in el:
                            f.write(s + '\n')
                    else:
                        f.write(el + '\n')
            else:
                f.write(v)

